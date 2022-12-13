# https://medium.com/@nutanbhogendrasharma/tensorflow-deep-learning-model-with-iris-dataset-8ec344c49f91

from sklearn import datasets
import tensorflow as tf
from carbontracker.tracker import CarbonTracker

# Step 1: Split the data into X and y
iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype('int64')  # 1 if Virginica, else 0

# Step 2: Define a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

# Step 3: Compile the model
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model and track with carbontracker
max_epochs = 5
tracker = CarbonTracker(epochs=max_epochs)

# Training loop.
for epoch in range(max_epochs):
    tracker.epoch_start()

    model.fit(X, y, batch_size=50)

    tracker.epoch_end()

tracker.stop()
