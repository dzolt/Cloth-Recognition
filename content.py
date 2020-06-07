import numpy as np
from utils import mnist_reader


X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
k_values = [i for i in range(150)]

def manhattan_distance(X, X_train):
    '''

    Zwraca odległość manhattana dla obektór ze zbioru X do obietków z X_train
    Dzielona przez 255 w celu optymalizacji wyników

    :param X: zbiór porównywanych obiektów
    :param X_train:  zbiór obiektów do których porównujemy
    :return: macierz odległości pomiędzy obiektami z X i X_train
    '''

    X = X.astype(int)
    X_train = X_train.astype(int)
    Dist = np.zeros(shape=(len(X), len(X_train)))

    for i in range(0, len(X)):
        Dist[i] = np.sum(abs(X_train - X[i]), axis=1)
        #print(Dist)

    return Dist

def sort_train_labels(Dist, y):
    """
    Sortowanie etykient klas danych treningowych y względem prawdopodobieństw w Dist

    :param Dist: macierz odległości pomiędzy obiektami X i X_train
    :param y: etykiety
    :return: Posortowana macierz etykiet klas posortowana względem wartości prawdopodobieństw
    """
    Dist = Dist.astype(int)
    y = y.astype(int)
    index_array = np.argsort(Dist, kind='mergesort', axis=1)
    return y[index_array]

def p_y_x_knn(y, k):
    """
    Wyznacza rozkład prawdopodobieństwa p(y|x) każdej z klas dla obietków ze zbioru
    testowego.

    :param y: macierz posortowanych etykiet dla danych treningowych
    :param k: liczba najbliższych sąsiadów
    :return: macierz prawdopodobieństw p(y|x) dla obietków z X
    """
    y = y.astype(int)
    unique_classes = np.sort(np.unique(y[0]))
    result = []
    for row in y:
        result.append([np.sum(row[0:k] == x) for x in unique_classes])
    return np.array(result) / k

def classification_error(p_y_x, y_true):
    """
    Wyznacza błąd klasyfikacji

    :param p_y_x: macierz przewidywanych prawdopodobieństw
    :param y_true: zbiór rzeczywistych etykiet klas
    :return: błąd klasyfikacji
    """
    p_y_x_reverse = np.fliplr(p_y_x)
    predict_labels = len(p_y_x[0]) - 1 - np.argmax(p_y_x_reverse, axis=1)
    result = np.mean(y_true != predict_labels)
    return result

def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicza błąd dla różnych wartości k. Następnie wyznacza najlepsze k, dla którego
    wartośc błędu jest najniższa

    :param X_val: zbiór danych walidacyjnych
    :param X_train: zbiór danych treningowych
    :param y_val: etykiety klas dla danych walidacyjnych
    :param y_train: etykiety klas dla danych treningowych
    :param k_values: wartości parametru k
    :return: krotke (best_accur, best_k), gdzie best_accur to wartość najlepszego dopasowania
    a best_k to wartość k dla której to dopasowanie wystąpiło
    """
    X_val = X_val.astype(int)
    X_train = X_train.astype(int)
    y_val = y_val.astype(int)
    y_train = y_train.astype(int)

    Dist = manhattan_distance(X_val, X_train)

    sorted_dist_matrix = sort_train_labels(Dist, y_train)

    errors = [classification_error(p_y_x_knn(sorted_dist_matrix, k), y_val) for k in k_values]

    best_error = np.amin(errors)
    best_k = k_values[np.argmin(errors)]
    print(f"Best error: {1 - best_error} \nBest k: {best_k}")
    return 1 - best_error, best_k



model_selection_knn(X_test, X_train, y_test, y_train, k_values)