from typing import Dict

from pandas.io.xml import preprocess_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.losses import mean_absolute_error, mean_squared_error

from thesis_code.thesis.webapp.src.backend.data_load.data_handle import DataLoader
from thesis_code.thesis.webapp.src.backend.models._features import FEATURES


class LinRegModel:
    def __init__(self, modelin):
        self.df = DataLoader().get_data()
        self.model = modelin
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        #self.features = FEATURES
        self.target = "Station2_PM10"

    def preprocess_data(self):
        X = self.df[FEATURES]
        y = self.df[self.target]
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                                                random_state=42)
    def train_model(self):
        self.preprocess_data()
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def get_forecast(self) -> Dict[str, float]:
        y_pred = self.train_model()
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        result = {"test": self.y_test.tolist(), "pred": y_pred.tolist(), "mae": float(mae), "r2": float(r2), "mse": float(mse)}
        return result