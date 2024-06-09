import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, template_folder='template')

# Load the model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input from the form
    house_area = [float(x) for x in request.form.values()]
    feature = [np.array(house_area)]
    prediction = model.predict(feature)
    output = (prediction)  # Round the prediction for better readability

    return render_template("index.html", prediction_text=f"Predicted Rent: ${output}")

if __name__ == "__main__":
    app.run(debug=True)
