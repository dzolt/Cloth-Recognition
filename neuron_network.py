import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


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

# obrazy mają rozmiar 28x28 więc spłaszczamy je do 1x784
model.add(Flatten(input_shape=(28, 28)))

# 128 to liczba neuronów na które składa się funkcja
model.add(Dense(128, activation='sigmoid'))

# 10 to liczba neuronów równa liczbie klas
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    verbose=1,
                    batch_size=256,
                    validation_data=(X_val, y_val))