from pandas import DataFrame as df
import pandas as pd

df = pd.read_json('./data/12000.json')
csv = df.to_csv('./data/12000.csv', index=False)

print(csv)