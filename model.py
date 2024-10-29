import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

plt.imshow(train_images[0], cmap='gray')
plt.show()

model = models.Sequential()

# Adding layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Adding Batch Normalization
model.add(layers.BatchNormalization())

# Flatten the output before passing to dense layers
model.add(layers.Flatten())

# Adding a Dropout layer to reduce overfitting
model.add(layers.Dropout(0.5))

# Adding Dense layers
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
model.save('trained_model.keras')
