from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models.linear_reg import LinRegModel
from backend.app.model_unpacking import ModelUnpack
from backend.models.lstm_v2 import LstmTwo



class Gru(LinRegModel):
    def __init__(self):
        model_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/ml_model/gru_model.keras"
        model = ModelUnpack(model_path).get_model()
        self.features = EN_FEATURES
        self.base_model = BaseModelEnsemble()
        super().__init__(model)

    def prepare_data(self):
        return self.base_model.preprocess_data()

    def get_forecast(self):
        res = self.prepare_data()
        X_test_seq = res['x_test_seq']
        y_test_seq = res['y_test_seq']

        predictions = self.model.predict(X_test_seq).flatten()

        mse = mean_squared_error(y_test_seq, predictions)
        mae = mean_absolute_error(y_test_seq, predictions)
        r2 = r2_score(y_test_seq, predictions)

        result = {
            "test": y_test_seq.tolist(),
            "pred": predictions.tolist(),
            "mae": float(mae),
            "r2": float(r2),
            "mse": float(mse)
        }
        return result

if __name__ == '__main__':
    print(Gru().get_forecast())