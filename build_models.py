from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.metrics import FalseNegatives, FalsePositives, TrueNegatives, TruePositives

import tensorflow as tf

def create_cnn4(node, dropout, learning,pixel):
    cnn4 = Sequential([
        Conv2D(node, kernel_size =(3,3) , activation = 'relu', input_shape=(pixel, pixel, 3)),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout),
        Conv2D(node*2, kernel_size =(3,3) , activation = 'relu', input_shape=(pixel, pixel, 3)),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout),
        Conv2D(node*3, kernel_size =(3,3) , activation = 'relu', input_shape=(pixel, pixel, 3)),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout),
        Conv2D(node*4, kernel_size =(3,3) , activation = 'relu', input_shape=(pixel, pixel, 3)),
        MaxPooling2D(pool_size=2),
        BatchNormalization(),
        Dropout(dropout*4),
        Flatten(),
        Dense(node, activation = 'relu'),
        BatchNormalization(),
        Dropout(dropout),
        Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning)
    cnn4.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', TruePositives(thresholds= .5), TrueNegatives(thresholds= .5), FalsePositives(thresholds= .5), FalseNegatives(thresholds= .5)])
    return cnn4