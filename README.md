# Cloth-Recognition
Made by Damian Żółtowski 246651 on 07.06.2020
  
  

## Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Celem zadania była implementacja modelu, który pozwoli
zakwalifikować zdjęcia reprezentujące ubrania.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Aby tego dokonać należało pobrać odpowiednie pliki 
zawierające dane testowe oraz walidacyjne z gita zalandoresearch. Następnie należało 
zadeklarować odpowiednie metody pozwalające stwierdzić do jakiej grupy ubrań zdjęcie ze 
zbioru walidacyjnego należy. W naszym przykładzie klasyfikujemy obiekty na 10 różnych
etykiet, którymi są:   
 * T-shirt/top
 * Trouser
 * Pullover
 * Dress
 * Coat
 * Sandal
 * Shirt
 * Sneaker
 * Bag
 * Ankle boot  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Naszymi danymi wejściowymi są ciągi testowe oraz
walidacyjne. Ciągi testowe zawierają 60.000 egzemplarzy każdy, przy czym ciąg X_train
zawiera obrazy w rozdzielczości 28x28 px. a y_train zawiera etykiety charakteryzujące 
dane. Ciągi walidacyjne natomiast mają 10.000 egzemplarzy danych o podobnej charakterystyce
co ciągi treningowe.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dane wyjściowe to dokładność z jaką nasz algorytm potrafił
będzie określić jaki przedmiot znajduje się na danym zdjęciu. Do tego zadania użyłem algorytmu 
modelu dyskryminującego K- Nearest Neighbours oraz posiłkując się materiałami dostępnymi na 
internecie zaprojektowałem sieć neuronową rozwiązującą problem klasyfikacyjny, wykorzystując
algorytm optymalizujący [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam), 
natomiast jako funkcję błędu wykorzystałem algorytm categorical crossentropy, który wylicza funkcję straty
między etykietami a ich predykacjami, używany tam gdzie jest więcej niż 2 etykiety, i który oczekuje, że
podane etykiety będą w tzw. reprezentacji one hot. Oznacza to, że etykiety będą reprezentowane binarnie,
a wektor y reprezentowany jako etykiety, np. [0, 1, 2, 3, 4], stanie się macierzą o następującym wyglądzie:
[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]. A więc pozycja 1 w 
rzędzie macierzy odpowiada do jakiej etykiety dany przedmiot będzie należał.

## Methods

## Results

## Usage


## Przypisy
[Zalando_github](https://github.com/zalandoresearch/fashion-mnist)
[Neural network tutorial](https://miroslawmamczur.pl/przykladowa-siec-neuronowa-mlp-w-tensorflow/?fbclid=IwAR35Pj0r1ul3cFH0RMnKZabto7Er0AfQN7vT9wfrbgq_RIm8ZnM3Ti_INaA)
[KNN image classification](https://medium.com/@YearsOfNoLight/intro-to-image-classification-with-knn-987bc112f0c2)
[KNN structure](https://www.ii.pwr.edu.pl/~zieba/zad2_msid.pdf)