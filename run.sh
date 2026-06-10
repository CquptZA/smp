python train_script.py \
    --train_feat_path ./feature_engineered_train.csv.gz \
    --test_feat_path ./feature_engineered_test.csv.gz \
    --drop_cols img_path \
    --seed 42 \
    --n_splits 5 \
    --tag v1