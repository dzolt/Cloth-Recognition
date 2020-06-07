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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Wszystkie ciągi zawierające obrazy zostały przekształcone na typ float
oraz podzielone na max. wartość pixela równą 255 w celu optymalizacji operacji matematycznych, ponieważ
wydaje się to być najefektywniejszy zakres liczbowy w jakim komputer może pracować.
## Methods

##### KNN ALGORITHM
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Czerpiąc wiedzę uzyskaną na laboratoriach numer dwa, postanowiłem zaimplementować pierwszy model dyskryminujący
jakim jest KNN. Jest to model nieparametryczny, czyli taki, dla którego parametrami modelu są dane uczące. 
Rozwiązanie tego problemu owym algorytmem sprowadza się do predykcji etykiety danego obrazu na podstawie, podobieństwa jakie ma ono z innymi obrazami.
W przypadku klasyfikacji testów i ich etykiet korzystaliśmy z metryki Hamminga, która opierała się na zwykłym porównaniu tekstów, czy słowo występuje,
czy też nie.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Tutaj natomiast nie możemy wykorzystać tej metody dlatego, że wartość pixela waha się w przedziale od 0 do 255.
Dlatego też jesteśmy zmuszeni użyć tutaj innej metryki. Można wykorzystać metrykę Euclidesową albo metrykę Manhatanna. Ta druga została zaimplementowana
w moim programie, a od Euclidesowej różni się jedynnie tym, że zamiast wartości bezwzględnej różnicy między pixelami, podnosimy je do kwadratu a następnie 
pierwiastkujemy.

![Manhattan_Distance](./pictures/Manhattan_Distance.PNG)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jest to kluczowa funkcja, która pomoże nam wyselekcjonować K najbliższych sąsiadów.
Następnie w funkcji **sort_train_labels()** sortujemy etykiety klas poprzez macierz odległości uzyskując posortowaną macierz etykiet obrazów.

![Sorted_ettiquets](./pictures/Sort_ett.PNG)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Po czym możemy policzyć prawdopodobieństwo p(y|x) wystąpienia etykiety y dla danego ciągu x. Czyli po prostu
dla każdej etykiety sumujemy ilość jej wystąpień w pierwszych k elementach każdego wierza macierzy y. Po czym dzielimy je na k otrzymując średnie
prawdopodobieństwa przynależności do poszczególnych klas. 

![P_Y_X](./pictures/p_y_x.PNG)
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Następnym krokiem było obliczenie błędu klasyfikacji, który polegał na sprawdzeniu ile obrazów zostało poprawnie
zakwalifikowanych przez nasz program jako odpowiadające im etykiety, przyrównując predykowane etykiety do zbioru walidacyjnego y_val.

![class_err](./pictures/class_err.PNG) 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Teraz należałó wybrać model, o najmniejszym błędzie klasyfikacji, przy róznych wartościach k.
Wykonałem klasyfikację modeli na podstawie ciągów walidacyjnych i testowych oraz ciągu zawierającego różną liczbę sąsiadów.
Funkcja zwraca najlepszą dokładność jaką udało się uzyskać oraz liczbę K sąsiadów, dla której owa wartość wystąpiła.
![select_mod](./pictures/mod_selec.PNG)
## Results

## Usage


## Przypisy
* [Zalando github](https://github.com/zalandoresearch/fashion-mnist)
* [Neural network tutorial](https://miroslawmamczur.pl/przykladowa-siec-neuronowa-mlp-w-tensorflow/?fbclid=IwAR35Pj0r1ul3cFH0RMnKZabto7Er0AfQN7vT9wfrbgq_RIm8ZnM3Ti_INaA)
* [KNN image classification](https://medium.com/@YearsOfNoLight/intro-to-image-classification-with-knn-987bc112f0c2)
* [KNN structure](https://www.ii.pwr.edu.pl/~zieba/zad2_msid.pdf)