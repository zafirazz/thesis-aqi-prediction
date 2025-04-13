from optparse import Option
from typing import Dict, Optional

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models.gbdt import Gbdt
from backend.models.linear_reg import LinRegModel


def lgbm_model():
    lgbm = LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    return lgbm

class Lgbm(Gbdt):
    def __init__(self):
        super().__init__()
        self.model = lgbm_model()
        self.base_model = BaseModelEnsemble()

    def prepare_data(self):
        return self.base_model.preprocess_data()

    def train_model(self, to_predict: Optional[np.ndarray] = None):
        if to_predict is None:
            res = self.base_model.create_test_train()
        else:
            res = self.base_model.create_test_train(to_predict)
        X_train_scaled = res['X train scaled']
        X_test_scaled = res['X test scaled']
        self.y_train = res['y train']

        self.model.fit(X_train_scaled, self.y_train)

        prediction = self.model.predict(X_test_scaled).flatten()
        return prediction

    def get_forecast(self, to_predict: Optional[np.ndarray] = None) -> Dict[str, float]:
        data = self.prepare_data()
        y_test_seq = data['y_test']

        if to_predict is None:
            predictions = self.train_model()
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
        else:
            predictions = self.model.predict(to_predict).flatten()

            result = {
                "pred": predictions.tolist()
            }

        return result