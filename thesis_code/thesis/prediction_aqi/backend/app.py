from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

from thesis_code.thesis.prediction_aqi.backend.linear_reg import LinearRegressor

app = Flask(__name__)

linear_regression_model = LinearRegressor()

@app.route('/models/linear_reg', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        return linear_regression_model.predict()
    elif request.method == 'GET':
        sample_payload = {
            "example_request": {
                "start_date": "2024-01-01",
                "endd_date": "2025-01-01",
            }
        }
        return jsonify(sample_payload)

if __name__ == '__main__':
    app.run(debug=True)