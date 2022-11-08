from abc import ABC

from typing import Optional, Tuple

from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)


class Detector(ABC):

	def __init__(self):

	def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        raise NotImplementedError
        
		


