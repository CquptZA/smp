import pandas as pd

df = pd.read_csv('feature_engineered_test.csv')
df.to_csv('feature_engineered_test.csv.gz', compression='gzip', index=False)

df = pd.read_csv('feature_engineered_train.csv')
df.to_csv('feature_engineered_train.csv.gz', compression='gzip', index=False)