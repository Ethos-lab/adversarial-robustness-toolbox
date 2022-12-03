# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
"""PyTest Module for testing blacklight detector"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pytest
import numpy as np
from art.utils import load_dataset
from art.defences.detector.evasion import BlacklightDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)


@pytest.fixture()
def get_mnist_data():
    """
    Get MNIST data and return {data, labels}
    """
    nb_test = 10
    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    y_test = np.argmax(y_test, axis=1)
    return x_test, y_test


@pytest.mark.only_with_platform("pytorch")
def test_blacklight_detector_pytorch_returns_nonempty_results(get_mnist_data):
    """
    Test black light detector to verify that it returns nonempty array
    """
    # Get MNIST
    x_test, _ = get_mnist_data

    blacklight_detector = BlacklightDetector(
        input_shape=x_test[0].shape, window_size=20, num_hashes_keep=50, num_rounds=50, step_size=1, workers=5
    )

    detected_results = blacklight_detector.detect(x=x_test[:100], threshold=25)
    assert detected_results


@pytest.mark.only_with_platform("pytorch")
def test_blacklight_detector_pytorch_returns_boolean_values(get_mnist_data):
    """
    Test black light detector to verify that it returns array with only 0 or 1
    """
    # Get MNIST
    x_test, _ = get_mnist_data

    blacklight_detector = BlacklightDetector(
        input_shape=x_test[0].shape, window_size=20, num_hashes_keep=50, num_rounds=50, step_size=1, workers=5
    )

    detected_results = blacklight_detector.detect(x=x_test[:100], threshold=25)
    assert all(detection in {0, 1} for detection in set(detected_results))
