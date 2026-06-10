import pandas as pd
import numpy as np
import itertools


from gensim.models import Word2Vec
import argparse
import os

import pickle
import joblib

# ------------------------------------------------------------
# 参数解析
parser = argparse.ArgumentParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

parser.add_argument('--input_feat', type=str, default=os.path.join(project_root, "post_feature.pkl"),
                    help='原始特征文件')

parser.add_argument('--model_store', type=str, default=os.path.join(project_root, "get_ebed"),
                    help='Word2Vec 模型保存/加载目录')

parser.add_argument('--output_emb', type=str, default=os.path.join(project_root, "user_embedding.pkl"),
                    help='输出的用户嵌入文件')
args = parser.parse_args()

os.makedirs(args.model_store, exist_ok=True)

# ------------------------------------------------------------
# 加载原始数据
with open(args.input_feat, "rb") as f:
    raw_data = pickle.load(f)

# 仅保留需要的字段（字段名不变，但内部使用不同别名）
keep_cols = ['Uid', 'top1_tags', 'top2_tags', 'top3_tags', 'Alltags', 'week_top1_tags']
raw_data = raw_data[keep_cols].copy()
print(f"数据形状: {raw_data.shape}")

# 获取唯一用户列表
unique_users = raw_data[['Uid']].drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------
# 核心函数：训练或加载 Word2Vec，并返回每个用户的平均嵌入向量
def train_or_load_w2v(dataframe, group_col, seq_col, model_index, mode_flag, 
                      embedding_dim=32, window_size=6, min_freq=5, skip_gram=1, model_dir=None):
    """
    mode_flag = 1 : 训练并保存模型
    mode_flag = 0 : 加载已有模型
    """
    # 按用户聚合标签列表（可能每个用户有多行标签，需要合并）
    grouped = dataframe.groupby(group_col, as_index=False)[seq_col].agg(
        lambda x: list(itertools.chain.from_iterable(x)) if isinstance(x.iloc[0], list) else list(x)
    )
    # 构造用于训练的词序列
    tag_sequences = grouped[seq_col].tolist()
    tag_sequences = [[str(item) for item in seq] for seq in tag_sequences]
    
    model_filename = os.path.join(model_dir, f"w2v_{model_index}.model")
    
    if mode_flag == 1:
        model = Word2Vec(
            sentences=tag_sequences,
            vector_size=embedding_dim,
            window=window_size,
            min_count=min_freq,
            sg=skip_gram,
            hs=0,
            seed=42
        )
        joblib.dump(model, model_filename)
        print(f"训练完成，模型已保存至: {model_filename}")
    elif mode_flag == 0:
        model = joblib.load(model_filename)
        print(f"加载已有模型: {model_filename}")
    else:
        raise ValueError("mode_flag 必须为 0（加载）或 1（训练）")
    
    # 为每个用户的标签序列生成平均嵌入向量
    def get_user_embedding(tag_list):
        vectors = [model.wv[word] for word in tag_list if word in model.wv]
        if len(vectors) == 0:
            return np.zeros(embedding_dim, dtype=np.float32)
        return np.mean(vectors, axis=0)
    
    emb_vectors = np.vstack([get_user_embedding(seq) for seq in tag_sequences])
    emb_df = pd.DataFrame(emb_vectors, columns=[f"emb_{model_index}_{i}" for i in range(embedding_dim)])
    emb_df[group_col] = grouped[group_col].values
    return emb_df

# ------------------------------------------------------------
# 配置训练模式（此处固定为训练模式，即 mode_flag = 1）
train_mode = 1   # 1=训练新模型, 0=加载已有模型
embedding_size = 32
model_counter = 1

# 待处理的标签列列表
tag_columns = ['Alltags', 'top1_tags', 'top2_tags', 'top3_tags']   # 可自行添加 week_top1_tags

# 存储所有嵌入特征的 DataFrame
user_embeddings = unique_users.copy()

for col in tag_columns:
    print(f"\n正在处理标签列: {col}")
    col_emb = train_or_load_w2v(
        dataframe=raw_data,
        group_col='Uid',
        seq_col=col,
        model_index=model_counter,
        mode_flag=train_mode,
        embedding_dim=embedding_size,
        window_size=6,
        min_freq=5,
        skip_gram=1,
        model_dir=args.model_store
    )
    # 合并到总表
    user_embeddings = user_embeddings.merge(col_emb, on='Uid', how='left')
    model_counter += 1

# 可选：处理 week_top1_tags
if 'week_top1_tags' in raw_data.columns:
    col = 'week_top1_tags'
    print(f"\n正在处理标签列: {col}")
    col_emb = train_or_load_w2v(
        dataframe=raw_data,
        group_col='Uid',
        seq_col=col,
        model_index=model_counter,
        mode_flag=train_mode,
        embedding_dim=embedding_size,
        window_size=6,
        min_freq=5,
        skip_gram=1,
        model_dir=args.model_store
    )
    user_embeddings = user_embeddings.merge(col_emb, on='Uid', how='left')
    model_counter += 1

# ------------------------------------------------------------
# 保存结果
user_embeddings.to_pickle(args.output_emb)
print(f"\n用户嵌入特征已保存至: {args.output_emb}")
print(f"最终形状: {user_embeddings.shape}")