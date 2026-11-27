import pandas as pd
from joblib import dump

# use linear regression
from sklearn.linear_model import LinearRegression

# split data into train and test sets
from sklearn.model_selection import train_test_split

# scores to check how good the model is
from sklearn.metrics import r2_score, mean_squared_error

# MLflow will track my results
import mlflow

df = pd.read_csv("data/bikes_cleaned.csv")
X = df[["age_years", "kms_ridden", "condition_score"]]
y = df["price"]

# split 80% to train and 20% to test and same split every time (42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
# train the model on the training data
model.fit(X_train, y_train)

# predict the test data
pred = model.predict(X_test)

# score R2 to see how well it fits
r2 = r2_score(y_test, pred)

#RMSE will give me the average error in euros
rmse = mean_squared_error(y_test, pred) ** 0.5

print(f"R2: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

# point MLflow to the server
mlflow.set_tracking_uri("http://localhost:5555")

# group all bike runs together 
mlflow.set_experiment("bike_price_predictor")

# start one run and save the settings and scores to MLflow
with mlflow.start_run():
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    
# save the trained model to a ocikle file
dump(model, "app/bike_model.pkl")
print("Saved app/bike_model.pkl")
