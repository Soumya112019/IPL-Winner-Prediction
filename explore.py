import pandas as pd

df = pd.read_csv("matches.csv")

print(df.head())
print("\nRows aur Columns:", df.shape)
print("\nColumns ke naam:", df.columns.tolist())

print("\nKhali jagah (Missing values):")
print(df.isnull().sum())

print("\nWinners ki ginti:")
print(df["winner"].value_counts())

feature_cols = ["city","team1","team2","toss_winner","toss_decision","venue"]
target_col = "winner"

X = df[feature_cols]
y = df[target_col]

print("\nFeatures (X) ki pehli 3 rows:")
print(X.head(3))

print("\nTarget (y) ki pehli 3 rows:")
print(y.head(3))
