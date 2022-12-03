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

from art.defences.detector.evasion.model_detector import ModelDetector
import sklearn.metrics.pairwise as pairwise
from collections import OrderedDict


class StatefulDefense(ModelDetector):
    """
    Implementation of the paper titled "Stateful Detection of Black-Box Adversarial Attacks"
    | Paper link: https://arxiv.org/pdf/1907.05587.pdf
    """

    def __init__(self, model, detector, K, threshold=None, training_data=None, chunk_size=1000, up_to_K=False):
        """
        Create a `StatefulDefense` instance which is used to the detect the presence of adversarial samples.

        :param model: The estimator model to be used for detection
        :param detector: The encoder needed to compute the detection
        :param K: Number of neighbors
        :param threshold: Limit for deciding if the input is clean or adversarial
        :param training_data: Input training data
        :param chunk_size: Size of chunks of input queries to be processed at once
        :param up_to_K: Boolean flag to decide if upper limit of K neighbors to be considered
        """
        self.K = K
        self.threshold = threshold
        self.training_data = training_data
        self.up_to_K = up_to_K

        super().__init__(model, detector)

        if self.threshold is None and self.training_data is None:
            raise ValueError("Must provide explicit detection threshold or training data to calculate threshold!")

        # super()._init_encoder(weights_path)
        if self.training_data is not None:
            print("Explicit threshold not provided...calculating threshold for K = %d" % K)
            _, self.thresholds = self.calculate_thresholds()
            self.threshold = self.thresholds[-1]
            print("K = %d; set threshold to: %f" % (K, self.threshold))

        self.num_queries = 0
        self.buffer = []
        self.memory = []
        self.chunk_size = chunk_size

        self.history = []  # Tracks number of queries (t) when attack was detected
        self.history_by_attack = []
        self.detected_dists = []  # Tracks knn-dist that was detected
        self.detections = []

    def detect(self, queries: np.ndarray) -> np.ndarray:
        """
        Perform the detections on the input queries and return boolean array.
        This returns True(1) if the input is adversarial, else return False(0) if clean.
        """
        self.process(queries)
        return self.detections

    def process(self, queries):
        """
        Encode the input queries and process each of them
        """
        queries = self.detector.encode(queries)
        for query in queries:
            self.process_query(query)

    def process_query(self, query):
        if len(self.memory) == 0 and len(self.buffer) < self.K:
            self.buffer.append(query)
            self.num_queries += 1
            self.detections.append(0)
            return False

        k = self.K
        all_dists = []

        if len(self.buffer) > 0:
            queries = np.stack(self.buffer, axis=0)
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        for queries in self.memory:
            dists = np.linalg.norm(queries - query, axis=-1)
            all_dists.append(dists)

        dists = np.concatenate(all_dists)
        k_nearest_dists = np.partition(dists, k - 1)[:k, None]
        k_avg_dist = np.mean(k_nearest_dists)

        self.buffer.append(query)
        self.num_queries += 1

        if len(self.buffer) >= self.chunk_size:
            self.memory.append(np.stack(self.buffer, axis=0))
            self.buffer = []

        is_attack = k_avg_dist < self.threshold
        if is_attack:
            self.detections.append(1)
            self.history.append(self.num_queries)
            self.detected_dists.append(k_avg_dist)
            self.clear_memory()
        else:
            self.detections.append(0)

    def clear_memory(self):
        """
        Emptying out the memory buffer
        """
        self.buffer = []
        self.memory = []

    def get_detections(self):
        history = self.history
        epochs = []
        for i in range(len(history) - 1):
            epochs.append(history[i + 1] - history[i])
        return epochs

    def calculate_thresholds(self, P=1000):
        """
        Compute thresholds for the given training data and number of neighbors
        """
        data = self.detector.encode(self.training_data)
        distances = []
        print(data.shape[0])
        for i in range(data.shape[0] // P):
            distance_mat = pairwise.pairwise_distances(data[i * P:(i + 1) * P,:], Y=data)
            distance_mat = np.sort(distance_mat, axis=-1)
            distance_mat_K = distance_mat[:,:self.K]
            distances.append(distance_mat_K)
        distance_matrix = np.concatenate(distances, axis=0)

        start = 0 if self.up_to_K else self.K

        THRESHOLDS = []
        K_S = []
        for k in range(start, self.K + 1):
            dist_to_k_neighbors = distance_matrix[:,:k + 1]
            avg_dist_to_k_neighbors = dist_to_k_neighbors.mean(axis=-1)
            threshold = np.percentile(avg_dist_to_k_neighbors, 0.1)

            K_S.append(k)
            THRESHOLDS.append(threshold)

        return K_S, THRESHOLDS
