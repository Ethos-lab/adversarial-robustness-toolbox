# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

from art.defences.detector.evasion.detector import Detector


class ModelDetector(Detector):
    def __init__(self, model, detector="SimilarityDetector"):
      self.detector = detector
      self.model = model

    def _init_encoder(self, weights_path):
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

