from detector import Detector

class ModelDetector(Detector):

	def __init__(self, model, detector="SimilarityDetector"):
		self.detector = detector
		self.model = model

	def __init_encoder(self, weights_path):
		encoder = self.model
        encoder.load_weights(weights_path, by_name=True)
        self.encoder = encoder
        self.encode = lambda x : encoder.predict(x)

    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        raise NotImplementedError


    def fit(self):


    def loss_gradient(self):

