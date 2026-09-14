import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained linear regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML Template with Embedded Styles (Glassmorphism & Soft Shadows)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            --card-bg: rgba(255, 255, 255, 0.9);
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --text-dark: #1f2937;
            --text-muted: #6b7280;
            --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-inset: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2.5rem;
            width: 100%;
            max-width: 650px;
            box-shadow: var(--shadow-lg);
            border: 1px solid rgba(255, 255, 255, 0.8);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            color: var(--text-dark);
            font-size: 1.875rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        .header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 0.4rem;
        }

        .input-group input {
            padding: 0.75rem 1rem;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
            background: #f9fafb;
            font-size: 0.95rem;
            color: var(--text-dark);
            box-shadow: var(--shadow-inset);
            outline: none;
            transition: all 0.2s ease-in-out;
        }

        .input-group input:focus {
            border-color: var(--primary);
            background: #ffffff;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 0.875rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.4);
            transition: all 0.2s ease-in-out;
        }

        .btn-submit:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 15px -3px rgba(79, 70, 229, 0.4);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 2rem;
            padding: 1.25rem;
            background: #ffffff;
            border-radius: 12px;
            text-align: center;
            box-shadow: var(--shadow-md);
            border-left: 5px solid var(--primary);
            animation: fadeIn 0.4s ease-out;
        }

        .result-box h3 {
            font-size: 0.9rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .result-box p {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
            margin-top: 0.25rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Real Estate Valuation</h1>
        <p>Enter the property features below to predict estimated value</p>
    </div>

    <form method="POST" action="/predict" class="form-grid">
        <div class="input-group">
            <label for="Square_Footage">Square Footage</label>
            <input type="number" step="any" name="Square_Footage" id="Square_Footage" required placeholder="e.g. 2100">
        </div>
        <div class="input-group">
            <label for="Num_Bedrooms">Bedrooms</label>
            <input type="number" step="any" name="Num_Bedrooms" id="Num_Bedrooms" required placeholder="e.g. 3">
        </div>
        <div class="input-group">
            <label for="Num_Bathrooms">Bathrooms</label>
            <input type="number" step="any" name="Num_Bathrooms" id="Num_Bathrooms" required placeholder="e.g. 2">
        </div>
        <div class="input-group">
            <label for="Year_Built">Year Built</label>
            <input type="number" step="any" name="Year_Built" id="Year_Built" required placeholder="e.g. 2015">
        </div>
        <div class="input-group">
            <label for="Lot_Size">Lot Size (sq ft)</label>
            <input type="number" step="any" name="Lot_Size" id="Lot_Size" required placeholder="e.g. 8500">
        </div>
        <div class="input-group">
            <label for="Garage_Size">Garage Size (cars)</label>
            <input type="number" step="any" name="Garage_Size" id="Garage_Size" required placeholder="e.g. 2">
        </div>
        <div class="input-group" style="grid-column: 1 / -1;">
            <label for="Neighborhood_Quality">Neighborhood Quality Score</label>
            <input type="number" step="any" name="Neighborhood_Quality" id="Neighborhood_Quality" required placeholder="e.g. 8.5">
        </div>

        <button type="submit" class="btn-submit">Calculate Estimated Price</button>
    </form>

    {% if prediction_text %}
    <div class="result-box">
        <h3>Estimated Valuation</h3>
        <p>{{ prediction_text }}</p>
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
    try:
        # Extract features matching the model's feature_names_in_ array
        features = [
            float(request.form['Square_Footage']),
            float(request.form['Num_Bedrooms']),
            float(request.form['Num_Bathrooms']),
            float(request.form['Year_Built']),
            float(request.form['Lot_Size']),
            float(request.form['Garage_Size']),
            float(request.form['Neighborhood_Quality'])
        ]
        
        feature_names = ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 
                         'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
        
        input_df = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_df)[0]
        
        output = f"${prediction:,.2f}"
        return render_template_string(HTML_TEMPLATE, prediction_text=output)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
