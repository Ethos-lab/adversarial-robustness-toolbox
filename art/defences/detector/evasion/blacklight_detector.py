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
"""
Module for blacklight detector implementation in ART
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from multiprocessing import Pool

import logging
from typing import Tuple
import numpy as np

from art.defences.detector.evasion.detector import Detector

logger = logging.getLogger(__name__)


def apply_hash(arguments):
    """
    Return the hash value for the image argument using SHA256 Algorithm
    """
    import hashlib

    img = arguments["img"]
    idx = arguments["idx"]
    window_size = arguments["window_size"]
    return hashlib.sha256(img[idx : idx + window_size]).hexdigest()


class BlacklightDetector(Detector):
    """
    Implementation of the paper "Blacklight: Scalable Defense for Neural Networks
    against Query-Based Black-Box Attacks" at USENIX Security 2022.
    Paper link: https://www.cs.cmu.edu/~neill/papers/mcfowland13a.pdf
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        window_size: int,
        num_hashes_keep: int,
        num_rounds: int = 50,
        step_size: int = 1,
        workers: int = 5,
        salt=None,
        **kwargs
    ):
        """
        Create a `BlacklightDetector` instance which is used to the detect the presence of adversarial samples.

        :param input_shape: The shape of the inputs to the detector
        :param window_size: The size of sliding window used for computing content hashes
        :param num_hashes_keep: Small set of hash entries used as probabilistic fingerprint
        :param num_rounds: Number of rounds for computing the hashes
        :param workers: Size of worker threads pools to run the detector
        """
        super().__init__()
        self.input_shape = input_shape
        self.window_size = window_size
        self.num_hashes_keep = num_hashes_keep
        self.num_rounds = num_rounds
        self.step_size = step_size
        self.workers = workers
        self.hash_dict = {}  # type: ignore
        self.output = {}  # type: ignore
        self.input_idx = 0
        self.pool = Pool(processes=workers)  # pylint: disable=R1732
        if salt is not None:
            self.salt = salt
        else:
            self.salt = np.random.rand(*self.input_shape) * 255.0

    def preprocess(self, array, num_rounds=1, normalized=True):
        """
        Preprocessing the array of inputs
        """
        if normalized:  # input image normalized to [0,1]
            array = np.array(array) * 255.0
        array = (array + self.salt) % 255.0
        array = array.reshape(-1)
        array = np.around(array / num_rounds, decimals=0) * num_rounds
        array = array.astype(np.int16)
        return array

    def hash_image(self, img, preprocess=True):
        """
        Compute the hash of an image after preprocessing
        """
        if preprocess:
            img = self.preprocess(img, self.num_rounds)
        total_len = int(len(img))
        idx_ls = []
        for i in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": i * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(apply_hash, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list

    def check_image(self, hashes):
        """
        Check if the image is already in the hashes computed, if yes, return the count
        """
        from collections import Counter

        sets = list(map(self.hash_dict.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return 0
        sets = Counter(sets)
        cnt = sets.most_common(1)[0][1]
        return cnt

    def detect_image(self, img):
        """
        Detect and find the number of occurrences for an input image
        """
        hashes = self.hash_image(img)[: self.num_hashes_keep]
        cnt = self.check_image(hashes)
        for i in hashes:
            if i not in self.hash_dict:
                self.hash_dict[i] = [self.input_idx]
            else:
                self.hash_dict[i].append(self.input_idx)
        return cnt

    def detect(self, x: np.ndarray, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Return an array of detections on the input queries. This returns True(1) if the
        count is greater than the threshold, else return False(0)
        """
        threshold = kwargs.get("threshold")
        detected_output = []
        for query in x:
            self.input_idx += 1
            detect_count = self.detect_image(query)
            if detect_count > threshold:
                detected_output.append(1)
            else:
                detected_output.append(0)
        return np.asarray(detected_output)
