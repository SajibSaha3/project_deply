from flask import Flask, request, render_template
import pickle
from helper import preprocess, query_point_creator

app = Flask(__name__, template_folder="template")

# Load the XGBoost model
with open("models/xgb_with_test.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", result="")

@app.route("/predict", methods=["POST"])
def predict():
    question1 = request.form["question1"]
    question2 = request.form["question2"]

    # Preprocess input questions and create query point
    query_point = query_point_creator(question1, question2)

    # Make prediction using the XGBoost model
    prediction = model.predict(query_point)

    # Interpret prediction result
    result = "Duplicate" if prediction[0] > 0.5 else "Not Duplicate"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

