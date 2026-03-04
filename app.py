from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all input values from form
        features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_features = np.array([features])
        
        # Make prediction
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "⚠️ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Invalid Input. Please enter valid numbers.")


if __name__ == "__main__":
    app.run(debug=True)