import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

num_classes = 10
input_shape = (28,28,1)

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

print("x_train shape:",x_train.shape)
print(x_train.shape[0],"train samples")
print(x_test.shape[0],"test samples")

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)
print(y_test.shape)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32,kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation="softmax"),
    ]
)
model.summary()


batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test,y_test,verbose=0)
print("Test loss:",score[0])
print("Test accuracy:",score[1])

def predict_function(n):
  plt.figure()
  plt.imshow(x_test[n],cmap="gray")
  plt.show()
  test_data = x_test[n].reshape(1,28,28,1)
  predict = model.predict(test_data)
  return np.argmax(predict)

print("predict: ",predict_function(0))