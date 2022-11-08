from detector import Detector
import numpy as np


class ModelDetector(Detector):
    def __init__(self, model, detector="SimilarityDetector"):
      self.detector = detector
      self.model = model

    def _init_encoder(self, weights_path):
        print("*")
        encoder = self.model
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(x)

    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        raise NotImplementedError


    def fit(self):
        raise NotImplementedError


    def loss_gradient(self):
        raise NotImplementedError

