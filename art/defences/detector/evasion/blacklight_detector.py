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

from art.config import ART_NUMPY_DTYPE

from art.defences.detector.evasion.detector import Detector

if TYPE_CHECKING:
    # pylint: disable=C0412
    import hashlib
    from collections import Counter

logger = logging.getLogger(__name__)


class BlacklightDetector(Detector):
    def __init__(
        self,
        window_size: int,
        num_hashes_keep: int,
        round: int = 50,
        step_size:int = 1,
        workers:int = 5,
        salt=None,
        **kwargs
    ):
        super().__init__(
            window_size = window_size,
            num_hashes_keep = num_hashes_keep,
            round = round,
            step_size = step_size,
            workers = workers,
            salt = None,
            **kwargs
        )
    
    def apply_hash(arguments):
        img = arguments['img']
        idx = arguments['idx']
        window_size = arguments['window_size']
        return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    def preprocess(self, array, round=1, normalized=True):
        if(normalized): # input image normalized to [0,1]
            array = np.array(array) * 255.
        array = (array + self.salt) % 255.
        array = array.reshape(-1)
        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def hash_image(self, img, preprocess=True):
        if preprocess:
            img = self.preprocess(img, round)
        total_len = int(len(img))
        idx_ls = []
        for el in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(self.apply_hash, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list

    def check_image(self, hashes):
        sets = list(map(self.hash_dict.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return 0
        sets = Counter(sets)
        cnt = sets.most_common(1)[0][1]
        return cnt
    
    def detect_image(self, img):
        self.input_idx += 1
        hashes = self.hash_image(img)[:self.num_hashes_keep]
        cnt = self.check_image(hashes)
        for el in hashes:
            if el not in self.hash_dict:
                self.hash_dict[el] = [self.input_idx]
            else:
                self.hash_dict[el].append(self.input_idx)
        return cnt

    def detect(self, x: np.ndarray, threshold: int, **kwargs) -> np.ndarray:
        detected_output = []
        if(self.salt == None):
            self.salt = np.random.rand(*query.shape) * 255.

        for query in x:
            detect_count = self.detect_image(query)
            if(detect_count > threshold):
                detected_output.append(1)
            else:
                detected_output.append(0)

        return detected_output