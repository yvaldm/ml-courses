"""

Perceptron training example from stepik.org

In this example we need to train two layer NN to perform XOR operation


"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = keras.Sequential([
    keras.layers.Dense(2, activation=tf.math.sigmoid),
    keras.layers.Dense(1, activation=tf.math.sigmoid)
])

model.compile(optimizer=tf.train.GradientDescentOptimizer(1),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=5000)

print(model.get_weights())
