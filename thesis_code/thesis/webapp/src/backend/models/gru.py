from typing import Dict, Optional

import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.data_load.data_handle import DataLoader
from backend.models.linear_reg import LinRegModel
from backend.app.model_unpacking import ModelUnpack
from backend.models.lstm_v2 import LstmTwo



class Gru:
    def __init__(self):
        model_path = "/Users/zafiraibraeva/Code/uni coding/thesis/thesis_code/thesis/webapp/ml_model/gru_model.keras"
        self.model = ModelUnpack(model_path).get_model()
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mae'])
        self.features = EN_FEATURES
        self.base_model = BaseModelEnsemble()
        self.features = EN_FEATURES
        self.target = "Station2_PM10"
        self.base_model = BaseModelEnsemble()
        self.df = DataLoader().get_data()
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None


    def prepare_data(self):
        return self.base_model.preprocess_data()

    def train_model(self):
        res = self.prepare_data()
        X_train_seq = res['x_train_seq']
        y_train_seq = res['y_train_seq']

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        print("[INFO] Training GRU model...")
        self.model.fit(
            X_train_seq, y_train_seq,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

    def get_forecast(self, to_predict: Optional[np.ndarray] = None):
        res = self.prepare_data()
        X_test_seq = res['x_test_seq']
        y_test_seq = res['y_test_seq']

        if to_predict is None:
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
        else:
            predictions = self.model.predict(to_predict).flatten()
            result = {
                "pred": predictions.tolist()
            }
        return result

if __name__ == '__main__':
    print(Gru().get_forecast())