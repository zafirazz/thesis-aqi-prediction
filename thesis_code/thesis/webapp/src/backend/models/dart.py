from typing import Dict

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models.linear_reg import LinRegModel


def create_dart():
    dart = LGBMRegressor(
        boosting_type='dart',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,
        random_state=42,
        verbose=-1
    )

class Dart(LinRegModel):
    def __init__(self):
        self.features = EN_FEATURES
        model = create_dart()
        self.base_model = BaseModelEnsemble()
        super().__init__(model)


    def train_model(self):
        res = self.base_model.create_test_train()
        X_test_scaled = res['X test scaled']
        self.y_train = res['y train scaled']
        self.model.fit(X_test_scaled, self.y_train)
        prediction = self.model.predict(X_test_scaled).flatten()
        return prediction

    def get_forecast(self) -> Dict[str, float]:
        res = self.base_model.preprocess_data()
        y_test_seq = res['y_test_seq']
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
        return result