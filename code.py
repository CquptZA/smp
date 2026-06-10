import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr



def load_data(train_path, test_path):
    def read_file(path):
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.gz'):
            # 支持 .csv.gz 或 .gz 后缀
            return pd.read_csv(path, compression='gzip')
        else:
            return pd.read_csv(path)
    
    log(f"读取训练特征: {train_path}")
    train = read_file(train_path)
    log(f"训练集形状: {train.shape}")
    
    log(f"读取测试特征: {test_path}")
    test = read_file(test_path)
    log(f"测试集形状: {test.shape}")
    return train, test


def preprocess_features(train_data, test_data, ignore_cols):
    label_name = 'label'
    pid_name = 'Pid'
    # 需要跳过的额外列
    extra_skip = {'datetime', 'description', 'joinedDate2', 'All_tags_len'}
    skip_set = extra_skip.union(set(ignore_cols))
    skip_set.add(label_name)
    skip_set.add(pid_name)
    
    # 候选特征列
    common_cols = [col for col in train_data.columns if col not in skip_set and col in test_data.columns]
    
    cat_feats = []
    # 遍历所有公共特征列
    for col in common_cols:
        # 处理 object 或 category 类型
        if train_data[col].dtype.name == 'category':
            # 已经是分类，检查测试集类别是否在训练集内
            cat_feats.append(col)
            train_cats = set(train_data[col].cat.categories)
            test_cats = set(test_data[col].cat.categories)
            if not test_cats.issubset(train_cats):
                # 将测试集超出部分转为 Missing
                test_data[col] = test_data[col].astype(str).fillna('Missing')
                test_data[col] = pd.Categorical(test_data[col], categories=train_data[col].cat.categories)
        elif train_data[col].dtype == 'object' or test_data[col].dtype == 'object':
            # 合并所有值并统一类别
            cat_feats.append(col)
            all_vals = pd.concat([train_data[col], test_data[col]]).astype(str).fillna('Missing')
            unique_cats = all_vals.unique()
            train_data[col] = pd.Categorical(train_data[col].astype(str).fillna('Missing'), categories=unique_cats)
            test_data[col] = pd.Categorical(test_data[col].astype(str).fillna('Missing'), categories=unique_cats)
    return train_data, test_data, common_cols, cat_feats


import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# ------------------------------------------------------------
def log(msg):
    print(f"[INFO] {msg}", flush=True)

# ------------------------------------------------------------
def load_data(train_path, test_path):
    def read_file(path):
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.gz'):
            # 支持 .csv.gz 或 .gz 后缀
            return pd.read_csv(path, compression='gzip')
        else:
            return pd.read_csv(path)
    
    log(f"读取训练特征: {train_path}")
    train = read_file(train_path)
    log(f"训练集形状: {train.shape}")
    
    log(f"读取测试特征: {test_path}")
    test = read_file(test_path)
    log(f"测试集形状: {test.shape}")
    return train, test

# ------------------------------------------------------------
def prepare_features(train, test, drop_cols):
    """准备特征列和标签，处理分类特征"""
    label = 'label'
    exclude_cols = {label, 'datetime', 'description', 'joinedDate2', 'All_tags_len', 'Pid'} | set(drop_cols)
    feats = [c for c in train.columns if c not in exclude_cols and c in test.columns]
    log(f"特征数量: {len(feats)}")
    
    cat_cols = []
    for col in feats:
        if train[col].dtype == 'object' or test[col].dtype == 'object':
            combined = pd.concat([train[col], test[col]], ignore_index=True).astype(str).fillna('Missing')
            cats = combined.unique()
            train[col] = pd.Categorical(train[col].astype(str).fillna('Missing'), categories=cats)
            test[col] = pd.Categorical(test[col].astype(str).fillna('Missing'), categories=cats)
            cat_cols.append(col)
        elif isinstance(train[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
            if not set(test[col].cat.categories).issubset(set(train[col].cat.categories)):
                test[col] = test[col].astype(str).fillna('Missing')
                test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)
    log(f"类别特征数量: {len(cat_cols)}")
    return train, test, feats, cat_cols

# ------------------------------------------------------------
def make_model(args, n_estimators=None):
    return lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        learning_rate=0.025,
        n_estimators=24000,
        colsample_bytree=0.3,
        colsample_bynode=0.7,
        extra_trees=True,
        random_state=int(args.seed),
    )

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_feat_path', required=True, help='训练集特征文件路径（parquet/csv）')
    parser.add_argument('--test_feat_path', required=True, help='测试集特征文件路径（parquet/csv）')
    parser.add_argument('--drop_cols', default='img_path', help='需要删除的列名，逗号分隔')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--tag', default='lgb_from_feat')
    args = parser.parse_args()

    # 固定输出目录和模型目录（已删除命令行参数）
    output_dir = './sub'
    model_dir = './model_save'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    train, test = load_data(args.train_feat_path, args.test_feat_path)

    drop_cols = [c.strip() for c in args.drop_cols.split(',') if c.strip()]
    train, test, feats, cat_cols = prepare_features(train, test, drop_cols)

    y = train['label'].to_numpy(dtype=np.float32)
    if 'Pid' in test.columns:
        post_ids = test['Pid'].map(lambda x: 'post' + str(x))
    else:
        post_ids = pd.Series([f'post{i}' for i in range(len(test))])

    # 交叉验证训练
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train), dtype=np.float32)
    test_preds = []
    fold_rows = []
    start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), start=1):
        log(f"========== fold {fold}/{args.n_splits} ==========")
        model = make_model(args)
        model.fit(
            train.iloc[tr_idx][feats], y[tr_idx],
            eval_set=[(train.iloc[va_idx][feats], y[va_idx])],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(200, verbose=False)],   # 原参数已删除，固定200轮
            categorical_feature=cat_cols if cat_cols else 'auto'
        )
        best_iter = getattr(model, 'best_iteration_', args.lgb_n_estimators)
        va_pred = model.predict(train.iloc[va_idx][feats])
        te_pred = model.predict(test[feats])
        oof[va_idx] = va_pred.astype(np.float32)
        test_preds.append(te_pred.astype(np.float32))

        mae = mean_absolute_error(y[va_idx], va_pred)
        sp = spearmanr(y[va_idx], va_pred).correlation
        fold_rows.append({'fold': fold, 'best_iteration': best_iter, 'mae': mae, 'spearman': sp})
        log(f"fold={fold} MAE={mae:.6f} Spearman={sp:.6f} elapsed={time.time()-start:.1f}s")

        model_path = os.path.join(model_dir, f'{args.tag}_fold{fold}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    oof_mae = mean_absolute_error(y, oof)
    oof_sp = spearmanr(y, oof).correlation
    log(f"OOF MAE={oof_mae:.6f} Spearman={oof_sp:.6f}")
    pd.DataFrame(fold_rows).to_csv(os.path.join(model_dir, f'{args.tag}_metrics.csv'), index=False)

    pred = np.mean(test_preds, axis=0)
    sub = pd.DataFrame({'post_id': post_ids.astype(str), 'popularity_score': pred})
    # 删除截断操作（原参数已删除）
    sub.to_csv(os.path.join(output_dir, f'sub_{args.tag}.csv'), index=False)
    log(f"提交文件已保存，统计: min={sub['popularity_score'].min():.4f}, max={sub['popularity_score'].max():.4f}")

# ------------------------------------------------------------
if __name__ == '__main__':
    main()



