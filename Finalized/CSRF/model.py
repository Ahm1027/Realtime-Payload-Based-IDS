import pandas as pd

df = pd.read_json('dataset.json')
print(df['data'])