from pathlib import Path
from tensorflow.keras.models import load_model
import joblib

class ModelUnpack:
    def __init__(self, model_path):
        self.model_path = model_path
        self.extension = Path(model_path).suffix

    def get_model(self):
        if self.extension in ['.keras', '.h5']:
            return load_model(self.model_path)
