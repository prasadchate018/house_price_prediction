import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load pickle model safely relative to current directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Property Valuation Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --accent: #10b981;
            --text-main: #0f172a;
            --text-sub: #64748b;
            --shadow-outer: 0 25px 50px -12px rgba(0, 0, 0, 0.45), 0 0 15px rgba(99, 102, 241, 0.15);
            --shadow-card: 0 10px 30px -5px rgba(0, 0, 0, 0.08);
            --shadow-input: inset 0 2px 4px 0 rgba(0, 0, 0, 0.04);
            --shadow-button: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .wrapper {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 2.75rem 2.5rem;
            width: 100%;
            max-width: 680px;
            box-shadow: var(--shadow-outer);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header {
            text-align: center;
            margin-bottom: 2.25rem;
        }

        .header h1 {
            color: var(--text-main);
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.03em;
        }

        .header p {
            color: var(--text-sub);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        .field-group {
            display: flex;
            flex-direction: column;
        }

        .field-group.full-width {
            grid-column: span 2;
        }

        .field-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .field-group input {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            font-size: 0.95rem;
            color: var(--text-main);
            box-shadow: var(--shadow-input);
            outline: none;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .field-group input:focus {
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15), var(--shadow-input);
        }

        .submit-btn {
            grid-column: span 2;
            margin-top: 1rem;
            padding: 1rem;
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: var(--shadow-button);
            transition: all 0.25s ease;
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.6);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .prediction-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 16px;
            text-align: center;
            box-shadow: var(--shadow-card);
            border-top: 4px solid var(--accent);
            animation: slideUp 0.35s ease-out forwards;
        }

        .prediction-card h3 {
            font-size: 0.85rem;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .prediction-card .price {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-main);
            margin-top: 0.25rem;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 580px) {
            .grid-form { grid-template-columns: 1fr; }
            .field-group.full-width { grid-column: span 1; }
            .submit-btn { grid-column: span 1; }
        }
    </style>
</head>
<body>

<div class="wrapper">
    <div class="header">
        <h1>Property Price Estimator</h1>
        <p>Predict residential property values using trained parameters</p>
    </div>

    <form method="POST" action="/predict" class="grid-form">
        <div class="field-group">
            <label for="Square_Footage">Square Footage</label>
            <input type="number" step="any" name="Square_Footage" id="Square_Footage" required placeholder="2500">
        </div>
        <div class="field-group">
            <label for="Num_Bedrooms">Bedrooms</label>
            <input type="number" step="any" name="Num_Bedrooms" id="Num_Bedrooms" required placeholder="4">
        </div>
        <div class="field-group">
            <label for="Num_Bathrooms">Bathrooms</label>
            <input type="number" step="any" name="Num_Bathrooms" id="Num_Bathrooms" required placeholder="3">
        </div>
        <div class="field-group">
            <label for="Year_Built">Year Built</label>
            <input type="number" step="any" name="Year_Built" id="Year_Built" required placeholder="2018">
        </div>
        <div class="field-group">
            <label for="Lot_Size">Lot Size (sq ft)</label>
            <input type="number" step="any" name="Lot_Size" id="Lot_Size" required placeholder="8000">
        </div>
        <div class="field-group">
            <label for="Garage_Size">Garage Size (cars)</label>
            <input type="number" step="any" name="Garage_Size" id="Garage_Size" required placeholder="2">
        </div>
        <div class="field-group full-width">
            <label for="Neighborhood_Quality">Neighborhood Quality (1-10)</label>
            <input type="number" step="any" name="Neighborhood_Quality" id="Neighborhood_Quality" required placeholder="8.5">
        </div>

        <button type="submit" class="submit-btn">Estimate Valuation</button>
    </form>

    {% if prediction_text %}
    <div class="prediction-card">
        <h3>Estimated Property Value</h3>
        <div class="price">{{ prediction_text }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: linear.pkl model file missing.")
    
    try:
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        feature_names = [
            'Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
            'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality'
        ]
        
        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_df)[0]
        
        formatted_price = f"${prediction:,.2f}"
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_price)
    
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

# Mandatory entry point for local debugging
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
