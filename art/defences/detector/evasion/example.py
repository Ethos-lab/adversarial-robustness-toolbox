from keras.datasets import cifar10
from stateful_defense import *

import numpy as np

def cifar10_encoder(encode_dim=256):
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

(x_train, _), (x_test, y_test) = cifar10.load_data()

x_test = x_test /  255.0
x_train = x_train / 255.0
perm = np.random.permutation(x_train.shape[0])

benign_queries = x_train[perm[:1000],:,:,:]
suspicious_queries = x_train[perm[-1],:,:,:] * np.random.normal(0, 0.05, (1000,) + x_train.shape[1:])

detector = StatefulDefense(model = cifar10_encoder(), detector = "SimilarityDetector", K = 50, threshold=None, training_data=x_train, chunk_size=1000, weights_path="./cifar_encoder.h5")

detector.process(benign_queries)

detections = detector.get_detections()
print("Num detections:", len(detections))
print("Queries per detection:", detections)
print("i-th query that caused detection:", detector.history)

detector.clear_memory()
detector.process(suspicious_queries)
detections = detector.get_detections()
print("Num detections:", len(detections))
print("Queries per detection:", detections)
print("i-th query that caused detection:", detector.history)
