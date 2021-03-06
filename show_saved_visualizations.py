# importy bibliotek
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# odczytujemy funkcje pomocnicze z odrębnego pliku, w którym przechowujemy
# je na wypadek, gdyby miały się nam przydać w innych programach
from ml_util_functions import *

# wyczyść ekran konsoli
os.system('cls')

# odczyt danych (zachowane w formacie binarnym nazywanym
# w Pythonie "pickle". Wykorzystujemy blok try-except aby
# obsłużyć przypadek, gdy nie zapisano żadnych danych, które
# mogłyby być odczytane
print('odczyt danych z pliku .dat')
try:
    indata = pd.read_pickle('parameterized_dataset.dat')
except:
    print()
    print('błąd odczytu pliku z danymi wizualizacji - czy zostały one zapisane na zakończenie wykonania skryptu run_softloss.py?')
    exit()

print('podgląd wczytanych danych:')
print(indata)
print()

# w Pandasie można wydzielać poszczególne kolumny tabeli za
# pomocą podania ich nazwy w nawiasach klamrowych, za pomocą selektora
# .loc można odczytać kolumny "od - do", w naszym przypadku np. zapis
# [:,'PCA_x':'PCA_z'] oznacza "wszystkie wiersze w kolumnach od PCA_x
# do PCA_y
people_list = indata['label_name'].to_numpy()
reduced_parameters_PCA = indata.loc[:,'PCA_x':'PCA_z'].to_numpy()

# Zestaw danych nie zawsze zawiera kolumny z punktami obliczonymi dla t-SNE i LDA
# Dzieje się tak, bo pełen komplet punktów mamy tylko wtedy gdy liczymy t-SNE
# bez redukcji rozmiaru zbioru, a to bardzo długo trwa. Dane DLA też możemy
# pozyskać jedynie przy liczbie klas większej lub równej 3
# Dlatego próbujemy dokonać odczytu w bloku try-except
tSNE_data_present = False
LDA_data_present  = False
try:
    reduced_parameters_tSNE = indata.loc[:,'tSNE_x':'tSNE_y'].to_numpy()
    tSNE_data_present = True
except:
    pass

try:
    reduced_parameters_LDA = indata.loc[:,'LDA_x':'LDA_z'].to_numpy()
    LDA_data_present  = True
except:
    reduced_parameters_LDA = np.array([])
    pass

# Parametry wydzielimy jako pierwsze 80 kolumn (indeksy od 0 do 79)
# Selekcję po numerach, a nie nazwach kolumn uzyskujemy w Pandasie tzw.
# selektorem, tutaj korzystamy z selektora iloc, który właśnie dokonuje
# selekcji po numerze kolumny
# Zapis [:,0:79] oznacza "wszystkie wiersze (:) z kolumn o numerach od 0 do 79"
parameters  = indata.iloc[:,0:79].to_numpy()

# Zapytajmy użytkownika, czy chce aby zmniejszyć ilość danych na wykresach 3D, to pomoże
# w przypadku zmiany widoku w okienku wizualizacji, duże zbiory danych "zacinają" się
# przy zmianie pozycji kamery.
print()
if ask_for_user_preference('Czy zwizualizować parametry w przestrzeni 3D?'):
    vis3D_reduced_parameters_PCA, vis3D_reduced_parameters_LDA, vis3D_people_list = ask_if_reduce_data_size([reduced_parameters_PCA,reduced_parameters_LDA, people_list], 'Czy zredukować rozmiar zbiorów danych na potrzeby wizualizacji 3D?\n(dla dużych zbiorów danych zmiana widoku na wykresie może trwać długo)')

print()
print('wyświetlenie wyników')

# mając wszystkie dane - wyświetlamy je w graficznej postaci za pomocą matplotliba
make_labelmasked_plot_2D(reduced_parameters_PCA, people_list, title='PCA',xlabel='1. komponent PCA',ylabel='2. komponent PCA')
if LDA_data_present:
    make_labelmasked_plot_2D(reduced_parameters_LDA, people_list, title='LDA',xlabel='1. komponent LDA',ylabel='2. komponent LDA')
if tSNE_data_present:
    make_labelmasked_plot_2D(reduced_parameters_tSNE, people_list, title='t-SNE',xlabel='1. komponent t-SNE',ylabel='2. komponent t-SNE')
try:
    make_labelmasked_plot_3D(vis3D_reduced_parameters_PCA, vis3D_people_list, title='PCA',xlabel='1. komponent PCA',ylabel='2. komponent PCA',zlabel='3. komponent PCA')
    if LDA_data_present:
        make_labelmasked_plot_3D(vis3D_reduced_parameters_LDA, vis3D_people_list, title='LDA',xlabel='1. komponent LDA',ylabel='2. komponent LDA',zlabel='3. komponent LDA')
except: pass

plt.show()