from typing import Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.app.model_unpacking import ModelUnpack
from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models._features import FEATURES
from backend.models.linear_reg import LinRegModel
import numpy as np

class LstmTwo(LinRegModel):
    def __init__(self):
        model_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/ml_model/lstm_model_v2.keras"
        model = ModelUnpack(model_path).get_model()
        self.features = EN_FEATURES
        self.base_model = BaseModelEnsemble()
        super().__init__(model)

    def prepare_data(self):
        return self.base_model.preprocess_data()

    def get_forecast(self) -> Dict[str, float]:
        res = self.prepare_data()
        X_test_seq = res['x_test_seq']
        y_test_seq = res['y_test_seq']

        predictions = self.model.predict(X_test_seq).flatten()

        lstm_mse = mean_squared_error(y_test_seq, predictions)
        lstm_mae = mean_absolute_error(y_test_seq, predictions)
        lstm_r2 = r2_score(y_test_seq, predictions)

        result = {
            "test": y_test_seq.tolist(),
            "pred": predictions.tolist(),
            "mae": float(lstm_mae),
            "r2": float(lstm_r2),
            "mse": float(lstm_mse)
        }
        return result

if __name__ == '__main__':
    m = LstmTwo()
    print(m.get_forecast())