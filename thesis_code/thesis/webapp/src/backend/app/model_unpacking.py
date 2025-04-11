from tensorflow.keras.models import load_model

class ModelUnpack:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(model_path, compile=False)

    def get_model(self):
        return self.model