from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

from thesis_code.thesis.prediction_aqi.backend.data import DataHandler


class LinearRegressor:
    def __init__(self):
        self.model = joblib.load("/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/prediction_aqi/models/pm10_prediction_model.pkl")
        self.data_handler = DataHandler()

    def predict(self):
        data = request.json
        start_date = data['start_date']
        end_date = data['end_date']

        filtered_df = self.data_handler.filter_by_date(start_date, end_date)

        if filtered_df.empty:
            return jsonify({'error': 'No data available for the given date rannge'}), 400

        features = self.data_handler.get_features(filtered_df)
        target = self.data_handler.get_target(filtered_df)
        predictions = self.model.predict(features)

        response = {
            'predictions': predictions.tolist(),
            'dates': filtered_df.index.strftime('%Y-%m-%d').tolist(),
            'features_used': features,
            'target': target,
        }

        return jsonify(response)
