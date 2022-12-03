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
"""PyTest Module for testing stateful defense detector"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pytest
import numpy as np
from art.utils import load_dataset
from art.defences.detector.evasion.stateful_defense import StatefulDefense
from art.defences.detector.evasion.encoder import SimilarityDetector

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)


@pytest.fixture()
def get_cifar10_data():
    """
    Get CIFAR10 data and return {data, labels}
    """
    nb_test = 100
    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    y_test = np.argmax(y_test, axis=1)
    return x_test, y_test


def cifar10_encoder(encode_dim=256):
    """
    Create cifar10 encoder model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, \
        MaxPooling2D, Activation

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', name='conv2d_1', input_shape=(32, 32, 3)))
    model.add(Activation('relu', name='activation_1'))
    model.add(Conv2D(32, (3, 3), name='conv2d_2'))
    model.add(Activation('relu', name='activation_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(64, (3, 3), padding='same', name='conv2d_3'))
    model.add(Activation('relu', name='activation_3'))
    model.add(Conv2D(64, (3, 3), name='conv2d_4'))
    model.add(Activation('relu', name='activation_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(512, name='dense_1'))
    model.add(Activation('relu', name='activation_5'))
    model.add(Dropout(0.5, name='dropout_3'))
    model.add(Dense(encode_dim, name='dense_encode'))
    model.add(Activation('linear', name='encoding'))

    return model


@pytest.mark.only_with_platform("tensorflow", "tensorflow2", "keras", "kerastf")
def test_stateful_detector_tensorflow_queries_returns_nonempty(get_cifar10_data):
    """
    Test stateful detector to return nonempty array
    """
    # Get CIFAR10
    x_test, _ = get_cifar10_data
    x_test = x_test / 255.0
    perm = np.random.permutation(x_test.shape[0])
    queries = x_test[perm[-1],:,:,:] * np.random.normal(0, 0.05, (1000,) + x_test.shape[1:])

    cifar10_model = cifar10_encoder()
    encoder = SimilarityDetector(model=cifar10_model)
    stateful_detector = StatefulDefense(
        model=cifar10_model,
        detector=encoder,
        k=50,
        threshold=None,
        training_data=x_test,
        chunk_size=1000
    )
    detections = stateful_detector.detect(queries)

    assert detections  # Non empty detections


@pytest.mark.only_with_platform("tensorflow", "tensorflow2", "keras", "kerastf")
def test_stateful_detector_tensorflow_queries_returns_boolean_values(get_cifar10_data):
    """
    Test stateful detector to return array with only 0 or 1
    """
    # Get CIFAR10
    x_test, _ = get_cifar10_data
    print(x_test.shape)
    x_test = x_test / 255.0
    perm = np.random.permutation(x_test.shape[0])
    queries = x_test[perm[-1],:,:,:] * np.random.normal(0, 0.05, (1000,) + x_test.shape[1:])

    cifar10_model = cifar10_encoder()
    encoder = SimilarityDetector(model=cifar10_model)
    stateful_detector = StatefulDefense(
        model=cifar10_model,
        detector=encoder,
        k=50, threshold=None,
        training_data=x_test,
        chunk_size=1000
    )
    detections = stateful_detector.detect(queries)

    assert all(detection in {0,1} for detection in set(detections))
