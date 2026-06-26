import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained GradientBoostingRegressor model
MODEL_PATH = "Gradient.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please ensure it's in the same directory.")

# HTML Template with UI styling built-in
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --panel-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: #0f172a;
            --border-rgba: rgba(255, 255, 255, 0.1);
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background: var(--bg-gradient); color: var(--text-main); min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 20px; }
        
        .container { width: 100%; max-width: 550px; background: var(--panel-bg); backdrop-filter: blur(16px); border: 1px solid var(--border-rgba); padding: 40px; border-radius: 24px; box-shadow: 0 20px 40px rgba(0,0,0,0.4); }
        h1 { font-size: 28px; font-weight: 700; text-align: center; margin-bottom: 8px; background: linear-gradient(to right, #fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { font-size: 14px; text-align: center; color: var(--text-muted); margin-bottom: 32px; }
        
        .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
        .full-width { grid-column: span 2; }
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        label { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
        
        input, select { background: var(--input-bg); border: 1px solid var(--border-rgba); padding: 12px 16px; border-radius: 12px; color: #fff; font-size: 15px; transition: all 0.2s ease; outline: none; }
        input:focus, select:focus { border-color: var(--accent-color); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); }
        
        button { background: var(--accent-color); color: #fff; border: none; padding: 14px; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; width: 100%; transition: background 0.2s ease, transform 0.1s ease; margin-top: 10px; }
        button:hover { background: var(--accent-hover); }
        button:active { transform: scale(0.98); }
        
        .result-box { margin-top: 32px; padding: 20px; background: rgba(99, 102, 241, 0.1); border: 1px dashed rgba(99, 102, 241, 0.3); border-radius: 16px; text-align: center; display: none; }
        .result-box.active { display: block; animation: fadeIn 0.4s ease forwards; }
        .result-val { font-size: 32px; font-weight: 700; color: #818cf8; margin-top: 4px; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @media (max-width: 480px) { .form-grid { grid-template-columns: 1fr; } .full-width { grid-column: span 1; } }
    </style>
</head>
<body>

<div class="container">
    <h1>Valuation Engine</h1>
    <div class="subtitle">Enter device specifications for precise valuation output</div>
    
    <form id="predictForm">
        <div class="form-grid">
            <div class="form-group full-width">
                <label for="brand">Brand (Categorical Mapping)</label>
                <select id="brand" name="Brand" required>
                    <option value="0">Asus</option>
                    <option value="1">Acer</option>
                    <option value="2">Lenovo</option>
                    <option value="3">HP</option>
                    <option value="4">Dell</option>
                </select>
            </div>
            <div class="form-group">
                <label for="speed">Processor Speed (GHz)</label>
                <input type="number" step="0.1" id="speed" name="Processor_Speed" placeholder="e.g. 2.5" required>
            </div>
            <div class="form-group">
                <label for="ram">RAM Size (GB)</label>
                <input type="number" id="ram" name="RAM_Size" placeholder="e.g. 16" required>
            </div>
            <div class="form-group">
                <label for="storage">Storage Capacity (GB)</label>
                <input type="number" id="storage" name="Storage_Capacity" placeholder="e.g. 512" required>
            </div>
            <div class="form-group">
                <label for="screen">Screen Size (Inches)</label>
                <input type="number" step="0.1" id="screen" name="Screen_Size" placeholder="e.g. 15.6" required>
            </div>
            <div class="form-group full-width">
                <label for="weight">Weight (kg)</label>
                <input type="number" step="0.01" id="weight" name="Weight" placeholder="e.g. 1.65" required>
            </div>
        </div>
        <button type="submit">Calculate Valuation</button>
    </form>

    <div class="result-box" id="resultBox">
        <label>Predicted Market Value</label>
        <div class="result-val" id="predictionValue">$0.00</div>
    </div>
</div>

<script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        const data = Object.fromEntries(formData);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            if (result.success) {
                document.getElementById('predictionValue').innerText = '$' + parseFloat(result.prediction).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                document.getElementById('resultBox').classList.add('active');
            } else {
                alert('Prediction Error: ' + result.error);
            }
        } catch (error) {
            alert('Server error occurred.');
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model pickle file not loaded on server.'}), 500
        
    try:
        input_data = request.json
        
        # Extract features ensuring exact order matching model training schema
        features = [
            float(input_data['Brand']),
            float(input_data['Processor_Speed']),
            float(input_data['RAM_Size']),
            float(input_data['Storage_Capacity']),
            float(input_data['Screen_Size']),
            float(input_data['Weight'])
        ]
        
        # Format for model inference shape (1, 6)
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]
        
        return jsonify({'success': True, 'prediction': float(prediction)})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
