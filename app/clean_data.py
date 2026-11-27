import pandas as pd
df = pd.read_csv("data/bikes.csv")

# show first 5 rows to check it loaded
print(df.head())

# remove spaces and make lowercase
df.columns = df.columns.str.strip().str.lower()

# drop any rows with missing values
df.dropna(inplace=True)

# drop duplicate rows
df.drop_duplicates(inplace=True)

# kms and age cannot be negative
df = df[(df["kms_ridden"] >= 0) & (df["age_years"] >= 0)]

# condition score must be between 1 and 5 
df = df[(df["condition_score"] >= 1) & (df["condition_score"] <= 5)]

# price cannot be negative
df = df[df["price"] > 0]

# save it
df.to_csv("data/bikes_cleaned.csv", index=False)


