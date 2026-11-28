# import flask for website, request gets form data and render_template shows html
from flask import Flask, request, render_template
from joblib import load

# create the Flask app and load model
app = Flask(__name__)
model = load("app/bike_model.pkl")


@app.route("/")
def home():
    # show index.html 
    return render_template("index.html")
    
# show predict page when user clicks predict
@app.route("/predict", methods=["POST"])
def predict():
    # get age, kms and condition from form and turn it into a number
    age = float(request.form["age_years"])
    kms = float(request.form["kms_ridden"])
    condition = float(request.form["condition_score"])
    
    # ask the model to predict 
    price = model.predict([[age, kms, condition]])[0]
    
    # show result.html with the price
    return render_template("result.html", price=round(price))
    
# run the app 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
