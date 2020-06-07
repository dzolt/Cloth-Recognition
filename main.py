from utils import mnist_reader
import content
import neuron_network

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_val, y_val = mnist_reader.load_mnist('data/fashion', kind='t10k')
k_values = [i for i in range(1, 150)]

# Wartości dzielone na 255 w celu optymalizacji działań
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

def main():
    print("=========================================================================\n")
    print("                              KNN PREDICTION                             \n")

    KNN_BEST_ACC, KNN_BEST_K = content.model_selection_knn(
                                X_val, X_train, y_val, y_train, k_values)

    print(f"KNN BEST ACCURACY: {KNN_BEST_ACC} \n KNN BEST K: {KNN_BEST_K}")

    print("=========================================================================\n")
    print("                              NEURON LEARNING PREDICTION                             \n")

    history = neuron_network.start(X_train, y_train, 50, 1, 60, 0.2)
    print("              DRAWING PLOTS TO SHOW ACCURACY:            \n")
    neuron_network.draw_curves(history=history)

    print("                        NEURON ACCURACY AND LOSS OUTPUT                            \n")
    neuron_network.evaluate(X_val, y_val, 32)


if __name__ == "__main__":
    main()