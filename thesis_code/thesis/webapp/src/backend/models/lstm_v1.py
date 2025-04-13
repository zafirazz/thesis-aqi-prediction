from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from backend.app.model_unpacking import ModelUnpack
from backend.models.linear_reg import LinRegModel

class LstmOne(LinRegModel):
    """Class for LSTM with 4 layers"""

    def __init__(self):
        model_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/ml_model/lstm_model_v1.keras"
        model = ModelUnpack(model_path).get_model()
        self.seq_length = 72
        self.features =  [
            "temp", "humidity", "precip",
            "precipcover", "cloudcover", "windspeed", "visibility",
            "winddir_sin", "winddir_cos", "is_heating_season", "is_work_day",
            "year", "month", "day"
        ]
        super().__init__(model)

    def create_sequences(self, data, target_index, seq_length):
        """
        Function to create sequences with input data and target.

        :param data: DataFrame with historical dataset
        :param target_index: PM10 target value for prediction
        :param seq_length: Length of sequence
        :return: two arrays with input data and target values.
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length, target_index])
        return np.array(X), np.array(y)

    def preprocess_data(self):
        """
        Functions that scales data and creates sequences.

        :return: none
        """
        data = self.df[self.features + [self.target]]
        df_scaled = self.scaler.fit_transform(data)
        X, y = self.create_sequences(df_scaled, target_index=len(self.features), seq_length=72)
        print(X.shape[2])
        split_idx = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]

    def inverse_transform_target(self, scaled_target):
        """
        Inverses scaled data into normal form.

        :param scaled_target: scaled PM10 values
        :return: inversed data
        """
        full_scaled = np.zeros((scaled_target.shape[0], len(self.features) + 1))  # +1 for target

        target_index = len(self.features)
        full_scaled[:, target_index] = scaled_target.flatten()

        inversed = self.scaler.inverse_transform(full_scaled)

        return inversed[:, target_index]

    def get_forecast(self) -> Dict[str, float]:
        """
        Prepares data for API post response.

        :return: dictionary with predicted, actual values and accuracy metrics
        """
        self.preprocess_data()

        predictions = self.model.predict(self.X_test)

        y_test_full = np.zeros((self.y_test.shape[0], len(self.features) + 1))
        y_test_full[:, len(self.features)] = self.y_test.flatten()
        y_test_actual = self.scaler.inverse_transform(y_test_full)[:, len(self.features)]

        y_pred_full = np.zeros((predictions.shape[0], len(self.features) + 1))
        y_pred_full[:, len(self.features)] = predictions.flatten()
        y_pred_actual = self.scaler.inverse_transform(y_pred_full)[:, len(self.features)]

        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)

        result = {
            "test": y_test_actual.tolist(),
            "pred": y_pred_actual.tolist(),
            "mae": float(mae),
            "r2": float(r2),
            "mse": float(mse)
        }

        return result


if __name__ == '__main__':
    m = LstmOne()
    print(m.get_forecast())