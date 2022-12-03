from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Step 1: Split the data into X and y
iris = datasets.load_iris()
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype('int64')  # 1 if Virginica, else 0

# Step 2: Convert X and Y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('X_train: ', X_train[0:5])
print('X_test: ', X_test[0:5])
print('y_train:', y_train[0:5])
print('y_test', y_test[0:5])

# Step 3: Define a model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

print('model: ', model)