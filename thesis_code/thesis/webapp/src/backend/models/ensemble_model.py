import numpy as np
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from backend.data_load.base_model_data import EN_FEATURES, BaseModelEnsemble
from backend.models.dart import Dart
from backend.models.gbdt import Gbdt
from backend.models.gru import Gru
from backend.models.lightgbm import Lgbm
from backend.models.lstm_v2 import LstmTwo


class Ensemble:
    """Class for Ensemble Model"""

    def __init__(self):
        self.features = EN_FEATURES
        self.target = "Station2_PM10"
        self.scaler = StandardScaler()
        self.split_data = BaseModelEnsemble()

        # Initialize base models
        self.base_models = {
            'gbdt': Gbdt(),
            'dart': Dart(),
            'lgbm': Lgbm(),
            'lstm': LstmTwo(),
            'gru': Gru()
        }

    def get_meta_features(self, X_tree, X_seq, models):
        """
        Generate meta-features for the ensemble model from base model predictions.

        :param X_tree: Input feature data for tree-based models (scaled)
        :param X_seq: Input sequence data for sequential models
        :param models: Base models for ensemble
        :return: meta_X : Stacked meta-feature matrix combining predictions from all base models.
        min_len : The minimum sequence length across all base model predictions,
        used for alignment of sequences.
        """
        tree_preds = {
            'gbdt': models['gbdt'].get_forecast(X_tree)["pred"],
            'dart': models['dart'].get_forecast(X_tree)["pred"],
            'lgbm': models['lgbm'].get_forecast(X_tree)["pred"]
        }

        seq_preds = {
            'lstm': models['lstm'].get_forecast(X_seq)["pred"],
            'gru': models['gru'].get_forecast(X_seq)["pred"]
        }

        min_len = min(len(p) for p in [*tree_preds.values(), *seq_preds.values()])

        meta_X = np.column_stack([
            tree_preds['gbdt'][-min_len:],
            tree_preds['dart'][-min_len:],
            tree_preds['lgbm'][-min_len:],
            seq_preds['lstm'][-min_len:],
            seq_preds['gru'][-min_len:]
        ])

        return meta_X, min_len

    def train_model(self):
        """
        Trains base models of meta-model and prepares meta-features.
        """
        time_steps = 30
        res = self.split_data.preprocess_data()

        X_train_scaled = res["x_train_scaled"]
        X_test_scaled = res["x_test_scaled"]
        X_train_seq = res["x_train_seq"]
        X_test_seq = res["x_test_seq"]
        y_train = res["y_train"]
        y_test = res["y_test"]

        print("Training base models...")
        for name, model in self.base_models.items():
            model.train_model()

        meta_X_train, min_len_train = self.get_meta_features(
            X_train_scaled[time_steps:], X_train_seq, self.base_models
        )
        meta_y_train = y_train[-min_len_train:]

        meta_X_test, min_len_test = self.get_meta_features(
            X_test_scaled[time_steps:], X_test_seq, self.base_models
        )
        meta_y_test = y_test[-min_len_test:]

        self.meta_X_train = meta_X_train
        self.meta_y_train = meta_y_train
        self.meta_X_test = meta_X_test
        self.meta_y_test = meta_y_test

        print(f"Meta-features training shape: {meta_X_train.shape}")
        print(f"Meta-features test shape: {meta_X_test.shape}")

    def ensemble_create(self):
        """
        Defines Ensemble meta-model architecture.

        :return: Compiled meta-model
        """
        input_dim = self.meta_X_train.shape[1]

        meta_model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        meta_model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='mse')

        return meta_model

    def get_forecast(self):
        """
        Trains final model

        :return: dictionary with predictions, test values and accuracy metrics result
        """
        self.train_model()

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        meta_model = self.ensemble_create()

        print("Training meta-model...")
        meta_model.fit(
            self.meta_X_train, self.meta_y_train,
            epochs=100,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )

        ensemble_pred = meta_model.predict(self.meta_X_test).flatten()

        mae = mean_absolute_error(self.meta_y_test, ensemble_pred)
        mse = mean_squared_error(self.meta_y_test, ensemble_pred)
        r2 = r2_score(self.meta_y_test, ensemble_pred)

        return {
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2),
            "test": self.meta_y_test.tolist(),
            "pred": ensemble_pred.tolist()
        }
