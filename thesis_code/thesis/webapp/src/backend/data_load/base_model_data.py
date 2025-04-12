from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.data_load.data_handle import DataLoader


EN_FEATURES = [
    "Station1_CO", "Station1_NO2", "Station1_NOx", "Station2_O3",
    "Station2_CO", "Station2_NO2", "Station2_NOx", "Station2_O3", "Station2_SO2",
    "Station1_SO2", "Station1_PM10",
    "temp", "humidity", "precip",
    "precipcover", "cloudcover", "windspeed", "visibility",
    "winddir_sin", "winddir_cos", "is_heating_season", "is_work_day",
    "year", "month", "day"
]


class BaseModelEnsemble:
    def __init__(self):
        self.df = DataLoader().get_data()
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.target = "Station2_PM10"
        self.features = EN_FEATURES

    def create_sequences(self, data, targets, time_steps=30):
        X_seq, y_seq = [], []
        for i in range(time_steps, len(data)):
            X_seq.append(data[i - time_steps:i, :])
            y_seq.append(targets[i])
        return np.array(X_seq), np.array(y_seq)

    def create_test_train(self) -> Dict[str, np.ndarray]:
        X = self.df[self.features].values
        y = self.df[self.target].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False)

        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        return {
            "X train scaled": X_train_scaled,
            "X test scaled": X_test_scaled,
            "y train": self.y_train,
        }

    def preprocess_data(self):
        time_steps = 30

        data_splits = self.create_test_train()

        X_train_scaled = data_splits['X train scaled']
        X_test_scaled = data_splits['X test scaled']
        self.X_train_seq, self.y_train_seq = self.create_sequences(X_train_scaled, self.y_train, time_steps)
        self.X_test_seq, self.y_test_seq = self.create_sequences(X_test_scaled, self.y_test, time_steps)

        return {
            "x_train_seq": self.X_train_seq,
            "x_test_seq": self.X_test_seq,
            "y_train_seq": self.y_train_seq,
            "y_test_seq": self.y_test_seq,
            "y_test": self.y_test
        }

if __name__ == '__main__':
    print(BaseModelEnsemble().preprocess_data())