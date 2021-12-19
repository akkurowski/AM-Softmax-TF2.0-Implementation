#------------------------------------------------------------------------------#
# Implementacja architektury neuronowej do uczenia nienadzorowanego,
# wykorzystująca straty rekonstrukcji (autoenkodera) i AM-Softmax.
#
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   19.12.2021
#------------------------------------------------------------------------------#

# importy bibliotek
import numpy as np
from scipy.io import wavfile
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from librosa.feature import mfcc
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from sklearn.manifold import TSNE
import csv
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

# odczytujemy funkcje pomocnicze z odrębnego pliku, w którym przechowujemy
# je na wypadek, gdyby miały się nam przydać w innych programach
from ml_util_functions import *

# wyczyść ekran konsoli
os.system('cls')

# Parametry wejściowe skryptu:
INPUT_DIRECTORY             = '_data_speech'
DATA_STORAGE_DIR            = '_stored_data'
PROGRESS_LOGGING_DIR        = '_progress_logs'
WEIGHTS_TRAINING_BACKUP_DIR = '_network_trn_backup'

# pozostałe nastawy pracy skryptu
SAMPLING_RATE      = 16_000 # [Sa/s]
FRAME_LENGTH       = int(200*0.001*SAMPLING_RATE) # [Sa]
MFCCGRAM_WIDTH     = 128
EXAMPLES_LIMIT     = 1000000000000000

# Utworzenie folderów, które muszą istnieć w trakcie wykonania dalszej części skryptu (w przypadku jeżeli ich nie ma)
# Z tego powodu można je po prostu usunąć jeśli potrzebny jest reset skryptu. Skrypt sam je ponownie utworzy i wypełni
# nową treścią.
if not os.path.isdir(DATA_STORAGE_DIR):os.mkdir(DATA_STORAGE_DIR)
if not os.path.isdir(PROGRESS_LOGGING_DIR):os.mkdir(PROGRESS_LOGGING_DIR)
if not os.path.isdir(WEIGHTS_TRAINING_BACKUP_DIR):os.mkdir(WEIGHTS_TRAINING_BACKUP_DIR)

# Blok zarządzania danymi wejściowymi - zależnie od preferencji użytkownika albo wykonywana jest świeża parametryzacja zbioru wejściowego
# (np. po dodaniu lub innej zmianie danych w zbiorze uczącym), albo odczyt wcześniej obliczonych współczynników (dużo szybsza opcja gdy te
# są dostępne).
if not ask_for_user_preference('Czy wczytać sparametryzowane MFCC-gramy z poprzednich uruchomień skryptu?'):
    
    list_of_examples = [] # Pusta lista, na którą będziemy przekazywać początki i końcówki utworów.
    people_list      = [] # Pusta lista, na którą będziemy przekazywać informację o tym, do kogo należy głos w każdej z ramek
    
    # Dla każego katalogu w folderze z danymi wejściowymi
    print('Wczytywanie danych...')
    
    # Odczyt przykładów z folderu wejściowego
    example_number = 0 # W razie potrzeby można ograniczyć liczbę wczytywanych danych - potrzebne jest do tego numerowanie kolejno wczytywanych przykładów
    for dir_name in os.listdir(INPUT_DIRECTORY):
        current_subfolder = os.path.join(INPUT_DIRECTORY,dir_name) # ścieżka do katalogu
        
        # jeżeli pod ścieżką current_subfolder faktycznie jest katalog, to
        if os.path.isdir(current_subfolder):
            # dla każdego pliku .wav w katalogu
            for file_name in os.listdir(current_subfolder):
                
                file_path = os.path.join(current_subfolder,file_name)
                if os.path.isdir(file_path): continue # pomijamy podfoldery - tam można "ukryć" pliki które nie mają być przetwarzane
                
                _, extension = os.path.splitext(file_name)
                
                # sprawdzenie poprawności formatu
                if extension != '.wav': raise RuntimeError('Wszystkie przetwarzane pliki muszą mieć format .wav.')
                
                # wczytanie pliku .wav
                warnings.filterwarnings("ignore") # Odczyt plików .wav często generuje warningi związane z tzw. "chunkami", które wpisywane są do plików niezgodnie ze specyfikacją standardu plikó WAVE, wyłączamy ostrzeżenia na czas odczytu bo jesteśmy tego faktu świadomi
                fs, audio_data = wavfile.read(file_path)
                warnings.filterwarnings("default")
                
                # sprawdzenie, czy jest ustawiona poprawna szybkość próbkowania
                if fs != SAMPLING_RATE: raise RuntimeError('Szybkość próbkowania pliku (%s) nie jest równa wartości nastawionej w skrypcie (%i)'%(file_name, SAMPLING_RATE))
                
                # Obliczamy na ile ramek jesteśmy w stanie podzielić przykład.
                chunk_length = len(audio_data)
                number_of_frames = chunk_length//FRAME_LENGTH
                
                # Dzielimy przykład na ranki i przekazujemy je do list (wraz z informacją o mówcach) 
                # do zdefiniowanych wcześniej pustych list.
                for frame_number in range(number_of_frames):
                    if example_number>=EXAMPLES_LIMIT:break # Przerywany przy osiągnięciu limitu
                    example_number += 1 # Każda ramka liczona jest jako "example"
                    audio_chunk_frame = audio_data[frame_number*FRAME_LENGTH:(frame_number+1)*FRAME_LENGTH] # Wydzielamy ramki
                    audio_chunk_frame = normalize(audio_chunk_frame) 
                    list_of_examples.append(audio_chunk_frame)
                    people_list += [dir_name]
                    
    # Pusta lista, na którą będziemy przekazywać mfcc-gramy początków i końcówek utworów.
    list_of_mfccgrams = []

    # dla każdego fragmentu danych audio z list_of_examples
    print(' ')
    print('parametryzowanie danych wejściowych')
    
    warnings.filterwarnings("ignore") # Obliczanie MFCC-gramów potrafi generować ostrzeżenia o tym, że mamy za dużo pasm melowych.
    # Są one puste, jednak MFCC-gram musi mieć określony rozmiar, by zgadzały się współczynniki poolingu w naszej sieci neuronowej.
    # Wiemy, że takie postępowanie wygeneruje warningi, które będą przeszkadzać w renderingu paska postępu parametryzacji, więc 
    # tu też wyłączymy ostrzeżenia
    for audio_data in tqdm(list_of_examples):
        
        n_fft = len(audio_data)//MFCCGRAM_WIDTH # Obliczenie długości przekształcenia FFT, która
        # wynika z wcześniejszych nastaw skryptu.
        
        # oblicz mfcc-gram
        mfcc_params = mfcc(audio_data, sr=SAMPLING_RATE, n_mfcc=40, n_fft=n_fft, hop_length=n_fft)
        
        # znormalizuj wartości mfcc-gramu do przedziału -1 do 1
        norm_factor = np.max(np.abs(mfcc_params))
        mfcc_params = mfcc_params/norm_factor*0.5
        
        # umieść mfccgram na liście wyników
        list_of_mfccgrams.append(mfcc_params)
    warnings.filterwarnings("default")

    # konwersja na macierz biblioteki NumPy
    array_of_mfccgrams  = np.array(list_of_mfccgrams)

    # obliczenie nowego kształtu danych wejściowych, który jest wymagany przez Kerasa
    # (dodanie liczby kanałów na końcu - 1 bo nasze spektrogramy są analogiem "czarno-białego obrazka")
    # docelowy kształt wejścia:
    # (liczba przykładów, liczba wsp. MFCC, liczba ramek MFCC, liczba kanałów - równa 1)
    old_mfccgrams_shape = list(array_of_mfccgrams.shape)
    new_mfccgrams_shape = tuple(old_mfccgrams_shape+[1])
    
    # zaaplikowanie nowego kształtu
    input_data = np.reshape(array_of_mfccgrams, new_mfccgrams_shape).astype(np.float32)
    people_list = np.array(people_list)
    
    print()
    # Użytkownik może zdecydować, czy chce zapisać otrzymane w tym kroku dane na później.
    if ask_for_user_preference('Czy zapisać sparametryzowane dane na dysku do późniejszego użycia?'):
        data_to_be_saved = {}
        data_to_be_saved.update({'list_of_examples':list_of_examples})
        data_to_be_saved.update({'people_list':people_list})
        data_to_be_saved.update({'input_data':input_data})
        np.save(os.path.join(DATA_STORAGE_DIR,'previous_parameterization_results.npy'),data_to_be_saved)
else:
    # Wykorzystujemy blok try-except aby obsłużyć przypadek, gdy nie zapisano żadnych danych, które
    # mogłyby być odczytane.
    try:
        retrieved_data_struct   = np.load(os.path.join(DATA_STORAGE_DIR,'previous_parameterization_results.npy'),allow_pickle=True).item()
        input_data              = retrieved_data_struct['input_data']
        people_list             = retrieved_data_struct['people_list']
    except:
        print()
        print('błąd odczytu sparametryzowanych danych - czy skrypt był już uprzednio wywoływany z poleceniem parametryzacji i zapisu danych do późniejszego użytku?')
        exit()

# generowanie etykiet do klasyfikacji
classes_as_str = np.unique(people_list)
NUM_LABELS     = len(classes_as_str)

# Dla wygory utworzymy słownik "tłumaczący" opisowe nazwy klasy (pseudonimy/imiona mówców)
# na identyfikujące ich wektory one-hot
labels_dict = {}
for i,class_str in enumerate(classes_as_str):
    labels_dict.update({class_str:gen_one_hot(i, NUM_LABELS)})

# Mając słownik definujący kto ma jaki wektor, przepisujemy listę "imion i nazwisk"
# na listę wektorów one-hot.
labels_as_onehot = []
for entry_str in people_list:
    labels_as_onehot.append(labels_dict[entry_str])
labels_as_onehot = np.array(labels_as_onehot)

print(' ')
print('tworzenie struktury sieci')

# import bibliotek do obsługi sieci neuronowych, TensorFlow ładuje się bardzo długo, więc importy uruchamiamy tuż przed tym,
# jak rzeczywiście są one potrzebne.
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,Input, Dense, UpSampling2D, MaxPooling2D, Flatten, Reshape, Add, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import PReLU
import tensorflow.keras.backend as K

# Jako kształt wejścia sieci potrzebujemy kształtu pojedynczego MFCC-gramu z liczbą kanałów równą 1
# (o co już wcześniej zadbaliśmy).
INPUT_SHAPE = input_data.shape[1::]

# Suma rozmiarów wektorów służy jako współczynnik korygujący stratę rekonstrukcji
# w algorytmie uczenia - potrzebne, jeżeli korzystamy z modułu autoenkodera.
TRN_REC_CORRECT_FCTR = 1
for elem in list(INPUT_SHAPE):
    TRN_REC_CORRECT_FCTR += elem

# -------------------------------------------------------------------------------------------
# DEFINICJA SIECI NEURONOWEJ

# Koder - ta podsieć przyjmuje MFCC-gram i zmienia go w wektor 80 wartości
encoder_input = Input(shape=INPUT_SHAPE)
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(encoder_input)
x             = MaxPooling2D((2,2))(x)
x             = Conv2D(32, (3,3), activation=PReLU(), padding='same')(x)
x_inc_out     = x # w ten sposób definiujemy "wyprzedzające", połączenia spinające długie bloku warstw
# taka sztuczka przyspiesza naukę w początkowych fazach treningu i nieco pomaga sieci w uczeniu się niektórych
# zależności algebraicznych, które mogą pojawić się w analizowanych danych
x             = Conv2D(32, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(32, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(32, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(32, (3,3), activation=PReLU(), padding='same')(x)
x             = Add()([x,x_inc_out]) # x_inc_out to początek wyprzedzającego połączenia w strukturze sieci,
# węzeł Add, to jego koniec - skopiowana wartość z początku bloku dodana jest do wyniku warstwy na jego końcu.
# czasami taki rodzaj połączenia nazywa się połączeniem rezydualnym (ang. residual connection).
x             = MaxPooling2D((2,2))(x)
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(x)
x_inc_out     = x
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(x)
x             = Conv2D(16, (3,3), activation=PReLU(), padding='same')(x)
x             = Add()([x,x_inc_out])
x             = MaxPooling2D((2,2))(x)
x             = Conv2D(1, (3,3), activation=PReLU(), padding='same')(x)
x             = Flatten()(x)
# Keras ma ograniczoną liczbę bloków - te których nie ma w nim domyślnie można szybko dogenerowywać za pomocą definicji w tzw. bloku
# lambda (analogia do tzw funkcji lambda:
# https://www.dummies.com/programming/python/how-to-use-lambda-functions-in-python/
# Dzięki temu nie trzeba mozolnie oficjalnie definiować nowej warstwy, tylko utworzyć warstwę lambda i za pomocą funkcji lambda "powiedzieć"
# jej co ma robić, jest to bardzo szybkie i wygodne, gdy warstwa ma robić jedną prostą rzecz, tak jak tutaj./
# To co robi lambda poniżej, to normalizacja wyjścia kodera tak, aby jego długość była równa 1, z "matematycznych" powodów
# przyspiesza to naukę w procesie uczenia się klastrów przez algorytm i jest też potrzebny, aby obliczenia funkcji straty AM-Softmax
# były stabilne numerycznie.
output        = Lambda(lambda x: K.l2_normalize(x,axis=1))(x) # normalizacja wektora wyjściowego tak, aby zawsze miał długość 1
encoder       = keras.Model(encoder_input, output, name="encoder")
encoder.summary()

# Klasa realizująca koder sieci uczącej się metryk dystansu, wykorzystujący stratę AM-Softmax
class SoftmaxDistanceMetricEncoder(keras.Model):
    def __init__(self, encoder, **kwargs):
        super(SoftmaxDistanceMetricEncoder, self).__init__(**kwargs)
        # Uchwyty do sieci kodera, dekodera
        self.encoder        = encoder
        
        # Dane i struktury straty AM-softmax
        self.enc_vec_width  = self.encoder.output_shape[1] # do obliczeń z góry musimy znać rozmiar wyjścia kodera.
        # konieczne jest także zdefiniowanie warstwy neuronów bez przesunięć (biasów), co można zareprezentować 
        # zwykłym tensorem (macierzą) o rozmiarze [długość_wektora_parametrów, liczba_klas].
        # Zdefiniujemy go w "surowym" TensorFlow, zmienną taką koniecznie należy zainicjalizować tzw. inicjalizatorem, jak poniżej:
        init_obj            = tf.keras.initializers.GlorotUniform()
        self.AM_sftmx_kernel= tf.Variable(  init_obj(shape=[self.enc_vec_width, NUM_LABELS]),
                                            name='AM_sftmx_kernel',
                                            shape=[self.enc_vec_width,NUM_LABELS],
                                            dtype=tf.float32)
    
    #algorytm treningu sieci
    def train_step(self, input_list):
        
        # rozdzielenie poszczególnych parametrów funkcji fit
        data   = input_list[0] # pierwszy parametr to dane wejściowe do sieci
        # są one także "odpowiedziami", których oczekujemy od rekonstruktora
        
        labels = input_list[1] # etykietki które posłużą jako odpowiedzi których
        # oczekujemy od klasyfikatora
        
        if isinstance(data, tuple):
            data = data[0]
        
        # teraz w kontekście GradientTape obliczymy stratę (loss) i gradienty
        # które optymalizator (optimizer) zaaplikuje w trakcie wykonania algorytmu
        # wstecznej propagacji błędów
        with tf.GradientTape() as tape:
            
            # wyliczmy odpowiedzi każdej z sieci (wewnątrz funkcji train_step) te odpowiedzi
            # to nie są zwykłe liczby, ale mają też dodatkowe fane pozwalające potem
            # optymalizatorowi wykonać algorytm wstecznej propagacji błędów
            encoded        = self.encoder(data)
            
            # Strata do klasteryzacji: AM-softmax, materiał źródłowy implementacji:
            # [1] https://arxiv.org/pdf/1801.05599.pdf
            # [2] https://towardsdatascience.com/additive-margin-softmax-loss-am-softmax-912e11ce1c6b
            # [3] https://medium.com/@rafayak/how-do-tensorflow-and-keras-implement-binary-classification-and-the-binary-cross-entropy-function-e9413826da7
            # [4] https://github.com/Joker316701882/Additive-Margin-Softmax
            
            # hiperparametry algorytmu AM-Softmax loss, ich działanie najlepiej zilustrowane jest w artykule
            # [1] na rysunku 4. Jest to kilka wybranych wartości wymienionych
            # w artykule, zawsze warto wypróbować, czy któraś z tych kombinacji działa lepiej niż inna.
            # s = 1; m = 0.35
            # s = 30; m = 0.35
            # s = 10; m = 0.20
            s = 10; m = 0.50
            
            # Zdefiniowaną wcześniej zmienną możemy nazywać jądrem (ang. kernel), stąd takie nazewnictwo zmiennych.
            # Dla stabilności numerycznej kernel też musi być znormalizowany tak aby długości wektorów odpowiadających
            # poszczególnym klasowm miały długość równą 1.
            AM_sftmx_kernel_norm = tf.math.l2_normalize(self.AM_sftmx_kernel, 0, 1e-10)
            
            # Tak jak w przytoczonym wcześniej artykule [1], po normalizacji wektora parametrów i jądra ich iloczyn skalarny (osiągany funkcją matmul)
            # daje nam tak naprawdę cos(theta), tak jak we wzorze (6)
            cos_theta = tf.linalg.matmul(encoded,AM_sftmx_kernel_norm)
            cos_theta = tf.clip_by_value(cos_theta, -1,1) # upewnijmy się, że wartości są prawidłowe i nie zafałszowały ich błędy numeryczne
            phi       = cos_theta - m # odejmujemy współczynnik marginesu, ta operacja wymusza separację pomiędzy klastrami
            AM_softmax_logits = s * tf.where(tf.equal(labels,1), phi, cos_theta) # Ta linijka podmienia współczynniki cos(theta) na cos(theta) - m zgodnie ze wzorem 
            # widocznym we wzorze (6), w zależności od klasy podmiana musi być w innym miejscu sumy i to nam zapewnia złożenie funkcji equal i where.
            
            # na koniec możemy policzyć już kros-entropię (z wbudowanym softmaxem - to akurat wygodna funkcjonalność)
            # więcej informacji na ten temat zawierają źródła [3] i [4]
            AM_sftmx_loss   = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=AM_softmax_logits)
            AM_sftmx_loss   = tf.reduce_mean(AM_sftmx_loss)
        
        # Po dokonaniu wszystkich obliczeń możemy nareszcie zaaplikować poprawki wynikające z gradientu obliczonej 
        # funkcji straty i zakończyć ten etap algorytmu treningu.
        grads = tape.gradient(AM_sftmx_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # zwracamy informację o tym, jakie to wykonanie train_step miało miary straty
        return {
            "val_loss": AM_sftmx_loss
        }
    
    # funkcja wykonywana gdy chcemy uruchomić funkcję predict() na obiekcie klasy SoftAE - ważna jeżeli
    # chcemy używać w praktyce naszego modelu. Zwracamy sobie wszystkie wyniki obliczeń z wewnątrz obiektu,
    # na jakich nam zależy
    def call(self, inputs):
        encoded = self.encoder(inputs)
        return encoded

# tworzymy obiekt klasy SoftmaxDistanceMetricEncoder, który przeprowadza proces nauki metryki dystansu (dokładniej - kodera sieci syjamskiej)
softmax_coder = SoftmaxDistanceMetricEncoder(encoder)

# Użytkownik może zdecydować, czy chce wczytać wagi z poprzedniego uruchomienia algorytmu, czy rozpocząć trening "od zera".
if ask_for_user_preference('Załadować ostatnie zachowane wagi sieci?'):
    # Wykorzystujemy blok try-except aby obsłużyć przypadek, gdy nie zapisano żadnych danych, które
    # mogłyby być odczytane.
    try:
        print('wagi zostaną przywrócone')
        softmax_coder.load_weights(os.path.join(DATA_STORAGE_DIR,'autoencoder_weights'))
    except:
        print()
        print('odczytywanie wag się nie powiodło - czy skrypt był już wywoływany i wydane zostało polecenie zapisu wag do późniejszego użytku?')
        exit()
else:
    print('trening zostanie rozpoczęty od zera')

# Wartość współczynnika nauki może być wybrana przez użytkownika z listy lub może być podana jej dokładna wartość
# po wybrani opcji "inna". W razie potrzeby trening może być też pominięty - np. gdy chcemy tylko przeliczyć wizualizacje
# i zapisać wyniki parametryzacji gotową już siecią.
print()
user_choice = ask_user_for_an_option_choice('Wybierz z listy poniżej współczynnik szybkości nauki (lr):', 'Numer wybranej opcji: ',[1e-3,1e-4,1e-5,'inna','pomiń trening'])
skip_training = False
if user_choice == 'pomiń trening':
    skip_training = True
elif user_choice == 'inna':
    LEARNING_RATE = ask_user_for_a_float('Podaj wartość współczynnika nauki: ')
else:
    LEARNING_RATE = user_choice

if skip_training:
    print()
    print('proces treningu zostanie pominięty')
    print('(nastąpi przejście do części skryptu odpowiedzialnej za wizualizację)')
    print()
else:
    print()
    print('Wybrano wartość lr = '+str(LEARNING_RATE))

    print()
    # tworzymy optymalizator i ustawiamy np. lr - czyli współczynnik prędkości nauki
    opt = optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    softmax_coder.compile(optimizer=opt) # kompilujemy model

    # Teraz utworzymy tzw. callbacki, czyli procedury uruchamiane przez Kerasa w trakcie procesu treningu.
    # Wykorzystamy kilka gotowych callbacków, które zapewnią nam autozapis wyników i możliwość śledzenia postępów
    # treningu w specjalnym narzędziu o nazwie TensorBoard.
    # Dokładne informacje odnośnie używania i tworzenia callbacków w Kerasie:
    # https://keras.io/api/callbacks/
    # https://keras.io/guides/writing_your_own_callbacks/

    # Ten callback specyfikuje, gdzie zapisywać dane ze śledzenia postępów treningu, jego przekazanie spowoduje, że 
    # będziemy mogli śledzić postępy treningu w panelu TensorBoarda, który uruchamia się pod adresem http://localhost:6006/
    # w przeglądarce internetowej - aby to zrobić trzeba uruchomić w osobnym okienku konsoli serwer TensorBoarda poleceniem:
    # tensorboard --logdir=<miejsce przechowywania danych treningu>
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(PROGRESS_LOGGING_DIR,generate_timestamp()))

    # Włąsny callback, który monitoruje funkcję straty i zapisuje model w momencie osiągnięcia najlepszego
    # wyniku zanotowanego w całej historii treningu
    class PeriodicWeightSaver(keras.callbacks.Callback):
        def __init__(self, saving_dir, file_name, interval):
            self.saving_dir = saving_dir
            self.interval   = interval
            self.file_name  = file_name
        
        def on_epoch_end(self,epoch,logs=None):
            if epoch%self.interval == 0:
                saving_fname = f"{self.file_name}_epoch_{epoch}_loss_{logs['val_loss']}"
                self.model.save_weights(os.path.join(self.saving_dir,saving_fname))
                

    # Włąsny callback, który monitoruje funkcję straty i zapisuje model w momencie osiągnięcia najlepszego
    # wyniku zanotowanego w całej historii treningu
    class BestSpecimenCallback(keras.callbacks.Callback):
        def __init__(self, saving_dir, file_name):
            self.best_loss  = None
            self.saving_dir = saving_dir
            self.file_name  = file_name
        
        def on_epoch_end(self,epoch,logs=None):
            if (self.best_loss is None) or (logs["val_loss"]<self.best_loss):
                self.best_loss = logs["val_loss"]
                
                saving_fname = f"{self.file_name}_epoch_{epoch}_loss_{logs['val_loss']}"
                self.model.save_weights(os.path.join(self.saving_dir,saving_fname))
                
                saving_fname = f"{self.file_name}"
                self.model.save_weights(os.path.join(self.saving_dir,saving_fname))
            
    
    run_timestamp                = generate_timestamp()
    best_specimen_bkp_name       = 'best_specimen_'+run_timestamp
    best_specimen_saver_callback = BestSpecimenCallback(WEIGHTS_TRAINING_BACKUP_DIR,best_specimen_bkp_name)
    
    # Wyświetlmy przypomnienie o tym, że trening można obserwować w TensorBoardzie
    print('\n\n')
    print('-----------------------------')
    print(' rozpoczęty zostanie proces treningu, pamiętaj, że możesz')
    print(' śledzić jego postępy w panelu narzędzia TensorBoard')
    print(' aby to zrobić, wpis w osobnym oknie konsoli poniższe polecenie:')
    print(' tensorboard --logdir="%s"'%(PROGRESS_LOGGING_DIR))
    print('-----------------------------')
    print('\n\n')

    # Uruchamiamy trening, jeśli osiągniemy limit to po prostu uruchamiamy go od nowa. 
    # Przechwytujemy wyjątek nazywany KeyboartInterrupt, który identyfikuje naciśnięcie kombinacji CTRL+C, co
    # pozwoli użytkownikowi manualnie przerywać skrypt, gdy uzna że chce zakończyć treningu, chce zobaczyć stan
    # algorytmu na wizualizacji PCA/t-SNE, lub gdy postanowi np. zmienić szybkość nauki (np. zmniejszyć ją, gdy
    # algorytm treningu zwolni po kilku pierwszych, "szybkich" epokach).
    try:
        while True:
            softmax_coder.fit(input_data,labels_as_onehot, epochs = 10_000, batch_size=64, callbacks=[tensorboard_callback, best_specimen_saver_callback]) # uruchamiamy trening
            
            print()
            print('ograniczenie epok osiągnięte, restart procedury treningowej')
            print()
            
    except KeyboardInterrupt:
        print('\n\ntrening przerwany ręcznie')
    last_specimen_bkp_name = 'last_training_weights'+run_timestamp
    softmax_coder.save_weights(os.path.join(WEIGHTS_TRAINING_BACKUP_DIR,last_specimen_bkp_name))

# Dodatkowo obejrzymy sobie przestrzeń punktów po redukcji jej z 80 wymiarów do 2
# wykorzystamy prostą metodę PCA, LDA oraz nieco bardziej zaawansowaną t-SNE
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# https://scikit-learn.org/0.16/modules/generated/sklearn.lda.LDA.html
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# Najpierw tworzymy obiekty wykonujące przekształcenia, w których ustawiamy ich parametry, np. to że chcemy wykonać rzutowanie na 2-wymiarową płaszczyznę
pca     = PCA(n_components=3) # PCA jest prosta i zachowuje maksimum wariancji oryginalnego zbioru, jednak możemy nie zaobserwować separacji klastrów które będą się nawzajem "przesłaniać" w wysokowymiarowej przestrzeni

# Dalsze kroku wymagają sprametryzowania zbioru wejściowego za pomocą kodera, który został właśnie przez nas wytrenowany.
parameters      = encoder.predict(input_data)

n_features           = parameters.shape[1]
n_classes            = len(np.unique(people_list))
lda_max_n_components = np.min([n_features, n_classes - 1])

# LDA jest podobna do PCA, ale uwzględnia informację o tym, który punkt należy do której klasy. Dzięki temu może obliczyć takie rzutowanie z przestrzeni wysokowymiarowej na niskowymiarową, że uzyskana jest maksymalna separacja pomiędzy klasami, przydatna do zgrubnego określenia wizualnie jak dobrze udało nam się rozseparować klasy.
# Niestety nie zawsze można wykorzystać LDA, gdyż jest ona aplikowalna jedynie gdy liczba komponentów wizualizowanych jest większa lub równa lda_max_n_components.
if lda_max_n_components<2:
    lda = None
elif lda_max_n_components==2:
    lda = LinearDiscriminantAnalysis(n_components=2)
else:
    lda = LinearDiscriminantAnalysis(n_components=3)

tSNE    = TSNE(n_components=2, n_jobs=8) # to przekształcenie zachowuje sąsiedztwo punktów, ale wszystkie inne relacje takie jak np. odległości między punktami są zniekształcone

print()

if lda is not None:
    print('wizualizacja rekonstrukcji i wyniku za pomocą PCA i LDA')
else:
    print('wizualizacja rekonstrukcji i wyniku za pomocą PCA')

# dokonujemy rzutowania poprzez wykonanie metody fit_transform()
reduced_parameters_PCA   = pca.fit_transform(parameters)

print()

if lda is not None:
    reduced_parameters_LDA   = lda.fit(parameters,people_list).transform(parameters)
else:
    reduced_parameters_LDA = np.array([])

# Rysujemy zobrazowania PCA i LDA
make_labelmasked_plot_2D(reduced_parameters_PCA, people_list, title='PCA',xlabel='1. komponent PCA',ylabel='2. komponent PCA')

if lda is not None:
    make_labelmasked_plot_2D(reduced_parameters_LDA, people_list, title='LDA',xlabel='1. komponent LDA',ylabel='2. komponent LDA')

plt.show() # pokazujemy efekty rekonstrukcji i wizualizacje

# Zapytajmy użytkownika, czy chce aby zmniejszyć ilość danych na wykresach 3D, to pomoże
# w przypadku zmiany widoku w okienku wizualizacji, duże zbiory danych "zacinają" się
# przy zmianie pozycji kamery.
print()
if ask_for_user_preference('Czy zwizualizować parametry w przestrzeni 3D?'):
    
    vis3D_reduced_parameters_PCA, vis3D_reduced_parameters_LDA, vis3D_people_list = ask_if_reduce_data_size([reduced_parameters_PCA,reduced_parameters_LDA, people_list], 'Czy zredukować rozmiar zbiorów danych na potrzeby wizualizacji 3D?\n(dla dużych zbiorów danych zmiana widoku na wykresie może trwać długo)')

    make_labelmasked_plot_3D(vis3D_reduced_parameters_PCA, vis3D_people_list, title='PCA',xlabel='1. komponent PCA',ylabel='2. komponent PCA',zlabel='3. komponent PCA')
    
    if lda is not None:
        make_labelmasked_plot_3D(vis3D_reduced_parameters_LDA, vis3D_people_list, title='LDA',xlabel='1. komponent LDA',ylabel='2. komponent LDA',zlabel='3. komponent LDA')
    plt.show()

# wizualizacja t-SNE jest bardzo kosztowna obliczeniowo, więc pytamy się użytkownika, czy ten chce ją uruchomić oraz czy chce, aby zmniejszyć
# rozmiar danych podlegających wizualizacji poprzez wybór co n-tego punktu danych. Pozwala to na przyspieszenie obliczeń, ale trzeba pamiętać, że
# bez kompletu danych nie będzie można zapisać współrzędnych osiągniętych przez t-SNE do zrzutu danych po parametryzacji i redukcji danych, jaki
# można wygenerować pod koniec wykonania skryptu. Na szczęście, nie zawsze jest to niezbędne, a samo t-SNE można też obliczyć później z zachowanych
# wektorów parametrów, które też są zapisywane do pliku.
print('')
if ask_for_user_preference('Zwizualizować wyniki za pomocą t-SNE? \n(obliczenia mogą być długie dla dużych zbiorów danych)'):
    
    tSNE_input_parameters, tSNE_people_list = ask_if_reduce_data_size([parameters, people_list], 'Czy zredukować rozmiar danych na wejściu t-SNE?')
    
    # po wykonaniu operacji przygotowawczych, można obliczyć już t-SNE
    reduced_parameters_tSNE  = tSNE.fit_transform(tSNE_input_parameters)
    
    # Rysujemy zobrazowanie w t-SNE
    make_labelmasked_plot_2D(reduced_parameters_tSNE, tSNE_people_list, title='t-SNE',xlabel='1. komponent t-SNE',ylabel='2. komponent t-SNE')
    plt.show()

# Użytkownik ma możliwość zapisania obecnych wag sieci do wydzielonego folderu w celu ich późniejszego przywrócenia.
# Jeśli coś pójdzie nie tak - ma możliwość ich niezapisania.
if not skip_training:
    print('')
    if ask_for_user_preference('Zapisać wagi do późniejszego użytku?'):
        choice = ask_user_for_an_option_choice('Który typ wag zapisać?','Wybrana opcja:',['najlepsze','ostatnie','pomiń zapis'])
        if not (choice == 'pomiń zapis'):
            if choice == 'najlepsze':
                try:
                    softmax_coder.load_weights(os.path.join(WEIGHTS_TRAINING_BACKUP_DIR,best_specimen_bkp_name))
                except:
                    print("Nie odnaleziono zapisu wag najlepszych, zapis zostanie wykonany z wykorzystaniem ostatnich wag (być może przez krótki trening nie wyodrębniono jeszcze najlepszych wag).")
            softmax_coder.save_weights(os.path.join(DATA_STORAGE_DIR,'autoencoder_weights'))
    else:
        print('wagi zostaną skasowane')

# Ostatecznie - mamy wektory parametrów, kompletne współrzędne PCA i w zależności od decyzji użytkownika także
# kompletne wspłrzędne t-SNE. Jeśli użytkonik sobie tego życzy - dokonujemy zapisu tych wartości na dysku za pomocą
# biblioteki Pandas.
print('')
if ask_for_user_preference('Zapisać sparametryzowany zbiór danych na dysku?'):
    output_parameters_df = pd.DataFrame(parameters)
    output_parameters_df = output_parameters_df.assign(label_name=people_list)
    
    # Zapiszmy współrzędne PCA i LDA.
    output_parameters_df = output_parameters_df.assign(PCA_x = reduced_parameters_PCA[:,0])
    output_parameters_df = output_parameters_df.assign(PCA_y = reduced_parameters_PCA[:,1])
    output_parameters_df = output_parameters_df.assign(PCA_z = reduced_parameters_PCA[:,2])
    
    if lda is not None:
        output_parameters_df = output_parameters_df.assign(LDA_x = reduced_parameters_LDA[:,0])
        output_parameters_df = output_parameters_df.assign(LDA_y = reduced_parameters_LDA[:,1])
        output_parameters_df = output_parameters_df.assign(LDA_z = reduced_parameters_LDA[:,2])
    
    # Pamiętajmy, że nie mamy gwarancji, że użytkownik zażyczył sobie wyliczenia współrzędnych t-SNE:
    try:
        # a nawet jeśli, to nie musiał obliczać t-SNE dla wszystkich punktów, niestety nie możemy w tak prosty 
        # sposób dodać danych do ramki DataFrame, jeżeli mamy puste miejsca w zapisie współczynników t-SNE, 
        # więc zapisu dokonujemy tylko, jeśli policzono komplet współczynników t-SNE.
        if reduced_parameters_tSNE.shape[0] == parameters.shape[0]: 
            output_parameters_df = output_parameters_df.assign(tSNE_x= reduced_parameters_tSNE[:,0])
            output_parameters_df = output_parameters_df.assign(tSNE_y= reduced_parameters_tSNE[:,1])
    except:
        pass
    
    # na koniec dokonujemy zapisu na dysk w binarny formacie pickle - to zapewni szybki odczyt i zapis danych,
    # plik będzie też mniejszy niż analogiczny plik w formacie tekstowym (np. .xlsx lub .csv).
    output_parameters_df.to_pickle('parameterized_dataset.dat')
