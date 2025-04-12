from typing import Dict

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from backend.app.model_unpacking import ModelUnpack
from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models.linear_reg import LinRegModel


def create_model():
    gbdt = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        random_state=42,
        verbose=1
    )
    return gbdt


class Gbdt(LinRegModel):
    def __init__(self):
        model = create_model()
        self.features = EN_FEATURES
        self.base_model = BaseModelEnsemble()
        super().__init__(model)

    def prepare_data(self):
        return self.base_model.preprocess_data()

    def train_model(self):
        res = self.base_model.create_test_train()
        X_train_scaled = res['X train scaled']
        X_test_scaled = res['X test scaled']
        self.y_train = res['y train']

        self.model.fit(X_train_scaled, self.y_train)

        prediction = self.model.predict(X_test_scaled).flatten()
        return prediction

    def get_forecast(self) -> Dict[str, float]:
        data = self.prepare_data()
        y_test_seq = data['y_test']

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
