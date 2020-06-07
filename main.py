from utils import mnist_reader

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
k_values = [i for i in range(1, 150)]

# Wartości dzielone na 255 w celu optymalizacji działań
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

