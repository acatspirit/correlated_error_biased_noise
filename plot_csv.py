import pandas as pd

l = 6
num_shots = 1000

file = f"l{l}_shots{num_shots}.csv"

data = pd.read_csv(file)

print(data.head())