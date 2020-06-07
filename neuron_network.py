import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt

### DATA PREPARE
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
          'Bag', 'Ankle boot']

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0


y_train = to_categorical(y_train, len(labels))
y_val = to_categorical(y_val, len(labels))
########################################################################################################

### MODEL PREPARE
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



def start(X_train, y_train, epchos, verbose, batch_size, validation_data):
    history = model.fit(X_train,
                        y_train,
                        epochs=50,
                        verbose=1,
                        batch_size=256,
                        validation_data=(X_val, y_val))
    return history

def draw_curves(history, key1='accuracy', ylim1=(0.8, 1.00),
                key2='loss', ylim2=(0.0, 1.0)):

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history[key1], "r--")
    plt.plot(history.history['val_' + key1], "g--")
    plt.ylabel(key1)
    plt.xlabel('Epoch')
    plt.ylim(ylim1)
    plt.legend(['train', 'test'], loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(history.history[key2], "r--")
    plt.plot(history.history['val_' + key2], "g--")
    plt.ylabel(key2)
    plt.xlabel('Epoch')
    plt.ylim(ylim2)
    plt.legend(['train', 'test'], loc='best')

    plt.show()


