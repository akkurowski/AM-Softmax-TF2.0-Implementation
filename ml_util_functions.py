#------------------------------------------------------------------------------#
# Funkcje pomocnicze dla skryptów realizujących zadania uczenia maszynowego
#
# author: Adam Kurowski
# mail:   akkurowski@gmail.com
# date:   25.08.2020
#------------------------------------------------------------------------------#
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

# Ten plik zawiera fragmenty kodu, które mogą być wykorzystywane przez
# wiele różnych skryptów - dzięki temu mamy pewność np. że wizualizacja będzie
# działać tak samo w skrypcie do treningu sieci, jak i w skrypcie
# który wyświetla same wizualizacje na podstawie zapisanych "na później"
# danych.

# niektóre z naszych zapisywanych plikó będziemy podpisywać ciągiem znaków
# identyfikującym czas wykonania skryptu - oto jak go będziemy generować:
def generate_timestamp():
    dateTimeObj = datetime.now() 
    timestampStr = dateTimeObj.strftime("%Y_%m_%d-%Hg%Mm%Ss")
    return timestampStr

# Dane audio warto znormalizować, aby przypadkowe różnice amplitudy nie zepsuł
# oczekiwanego przez nas wyniku działania programu
def normalize(input_audio):
    norm_value  = np.max(np.abs(input_audio))
    input_audio = input_audio/norm_value
    return input_audio

# Funkcja realizująca komunikację z użytkownikiem skryptu.
# Służy do zadania pytania z odpowiedzią tak/nie zwracaną w postaci
# zmiennej logicznej
def ask_for_user_preference(question):
    while True:
        user_input = input(question+' (t/n): ')
        if user_input == 't':
            return True
            break
        if user_input == 'n':
            return False
            break

# Funkcja służąca do komunikacji z użytkownikiem, która prosi o
# wybór jednek z kilku dostępnych opcji (poprzez podanie jej numeru).
def ask_user_for_an_option_choice(question, val_rec_prompt, items):
    print(question)
    allowed_numbers = []
    for it_num, item in enumerate(items):
        print('\t (%i)'%(it_num+1)+str(item))
        allowed_numbers.append(str(it_num+1))
    while True:
        user_input = input(val_rec_prompt)
        if user_input in allowed_numbers:
            return items[int(user_input)-1]

# Funkcja do komunikacji z użytkownikiem, która prosi o podanie
# przez użytkownika wartości zmiennoprzecinkowej.
def ask_user_for_a_float(question):
    while True:
        user_input = input(question)
        try:
            float_value = float(user_input)
            return float_value
            break
        except:
            pass

# Niektóre przekształcenia potrzebne do wizualizacji takie jak t-SNE liczą się bardzo długo,
# dlatego dobrze czasami jest zapytać użytkownika, czy zmniejszyć ilość danych wybierając co n-tą próbkę
def ask_if_reduce_data_size(vectors_for_reduction, question):
    print()
    if ask_for_user_preference(question):
        user_choice = ask_user_for_an_option_choice('Ilukrotnie należy zredukować rozmiar danych?', 'Numer wybranej opcji: ',[5,10,100,'inna'])
        if user_choice == 'inna':
            data_decim_rate = ask_user_for_a_float('Podaj współczynnik redukcji: ')
        else:
            data_decim_rate = user_choice
        # zmniejszamy rozmiar danych poprzez wybór co n-tej próbki.
        data_decim_rate = int(data_decim_rate)
        output_vectors = []
        for vec in vectors_for_reduction:
            output_vectors.append(vec[::data_decim_rate])
        return tuple(output_vectors)
    else:
        return tuple(vectors_for_reduction)

# Ten fragment kodu, który podpisuje osie wykresów, jest wspólny zarówno dla
# funkcji kreślących w 2D, jak i w 3D
def plot_handle_kwargs(kwargs):
    if 'title'  in kwargs.keys() is not None: plt.gca().set_title(kwargs['title'])
    if 'xlabel' in kwargs.keys() is not None: plt.gca().set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys() is not None: plt.gca().set_ylabel(kwargs['ylabel'])
    if 'zlabel' in kwargs.keys() is not None: plt.gca().set_zlabel(kwargs['zlabel'])

# Procedura generująca wykresy w 2D
def make_labelmasked_plot_2D(reduced_parameters, people_list, **kwargs):
    plt.figure()
    for person_name in np.unique(people_list):
        person_mask = (people_list == person_name)
        plt.scatter(reduced_parameters[person_mask,0],reduced_parameters[person_mask,1],label=person_name)
    plt.legend()
    plt.grid()
    plot_handle_kwargs(kwargs)

# Procedura generująca wykresy w 3D
def make_labelmasked_plot_3D(reduced_parameters, people_list, **kwargs):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    for person_name in np.unique(people_list):
        person_mask = (people_list == person_name)
        ax.scatter(reduced_parameters[person_mask,0],reduced_parameters[person_mask,1],reduced_parameters[person_mask,2],label=person_name)
    plt.legend()
    plt.grid()
    plot_handle_kwargs(kwargs)

# klasa w sieci neuronowej jest reprezentowana jako wektor one-hot, tzn. mając np. trzy klasy klasa 1.
# jest reprezentowana przez wektor [1 0 0], klasa 2. jako [0 1 0], a klasa 3. jako [0 0 1].
# Poniższa funkcja oblicza takie wektory mając informację o numerze klasy i liczbie klas (potrzebnej
# do ustalenia jak długi ma być wektor).
def gen_one_hot(class_num, NUM_LABELS):
    output = np.zeros(NUM_LABELS)
    output[class_num] = 1
    return output