from pathlib import Path
from tensorflow.keras.models import load_model
import joblib

class ModelUnpack:
    """
    Class to load saved keras models.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.extension = Path(model_path).suffix

    def get_model(self):
        """If model was saved as .keras or .h5 file, load it and return it."""
        if self.extension in ['.keras', '.h5']:
            return load_model(self.model_path)
