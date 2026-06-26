from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Gradient.pkl")
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# HTML Frontend Template (Embedded directly in the python file)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Predictor</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); width: 100%; max-width: 500px; }
        h2 { margin-top: 0; color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #555; font-size: 14px; }
        input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; box-sizing: border-box; font-size: 14px; }
        input:focus { border-color: #0070f3; outline: none; }
        button { width: 100%; padding: 12px; background-color: #0070f3; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; transition: background 0.2s; }
        button:hover { background-color: #0051a8; }
        .result-box { margin-top: 20px; padding: 15px; border-radius: 6px; background-color: #f0f7ff; border: 1px solid #cce3ff; color: #004085; text-align: center; font-weight: bold; display: none; }
    </style>
</head>
<body>

<div class="container">
    <h2>Model Prediction Form</h2>
    <form id="predictionForm">
        <div class="form-group">
            <label for="Brand">Brand (Numeric Code)</label>
            <input type="number" id="Brand" required value="1">
        </div>
        <div class="form-group">
            <label for="Processor_Speed">Processor Speed (GHz)</label>
            <input type="number" step="any" id="Processor_Speed" required value="2.5">
        </div>
        <div class="form-group">
            <label for="RAM_Size">RAM Size (GB)</label>
            <input type="number" id="RAM_Size" required value="8">
        </div>
        <div class="form-group">
            <label for="Storage_Capacity">Storage Capacity (GB)</label>
            <input type="number" id="Storage_Capacity" required value="512">
        </div>
        <div class="form-group">
            <label for="Screen_Size">Screen Size (Inches)</label>
            <input type="number" step="any" id="Screen_Size" required value="15.6">
        </div>
        <div class="form-group">
            <label for="Weight">Weight (kg)</label>
            <input type="number" step="any" id="Weight" required value="1.8">
        </div>
        <button type="submit">Predict Price</button>
    </form>

    <div id="result" class="result-box"></div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'none';

        // Gather data from inputs
        const payload = {
            Brand: parseFloat(document.getElementById('Brand').value),
            Processor_Speed: parseFloat(document.getElementById('Processor_Speed').value),
            RAM_Size: parseFloat(document.getElementById('RAM_Size').value),
            Storage_Capacity: parseFloat(document.getElementById('Storage_Capacity').value),
            Screen_Size: parseFloat(document.getElementById('Screen_Size').value),
            Weight: parseFloat(document.getElementById('Weight').value)
        };

        try {
            // Post payload to backend prediction API endpoint
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            
            if (result.status === "success") {
                resultDiv.style.backgroundColor = "#f0f7ff";
                resultDiv.style.borderColor = "#cce3ff";
                resultDiv.style.color = "#004085";
                resultDiv.innerText = `Predicted Value: ${result.prediction.toFixed(2)}`;
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            resultDiv.style.backgroundColor = "#fff3cd";
            resultDiv.style.borderColor = "#ffeeba";
            resultDiv.style.color = "#856404";
            resultDiv.innerText = `Error: ${error.message}`;
        }
        
        resultDiv.style.display = 'block';
    });
</script>

</body>
</html>
"""

# Frontend Route
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

# Backend API Endpoint Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Structure the features properly
        features = [
            float(data["Brand"]),
            float(data["Processor_Speed"]),
            float(data["RAM_Size"]),
            float(data["Storage_Capacity"]),
            float(data["Screen_Size"]),
            float(data["Weight"])
        ]
        
        input_array = np.array([features])
        prediction = model.predict(input_array)
        
        return jsonify({
            "status": "success",
            "prediction": float(prediction[0])
        })

    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
