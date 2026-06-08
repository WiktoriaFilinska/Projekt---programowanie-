# Przewidywanie cukrzycy — Klasyfikacja binarna

Projekt zaliczeniowy z przedmiotu Programowanie 2.
Celem projektu jest implementacja i porównanie modeli uczenia maszynowego
do przewidywania wystąpienia cukrzycy typu 2 na podstawie danych medycznych.

## Zbiór danych

**Nazwa:** Pima Indians Diabetes Database  
**Źródło:** [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Rozmiar:** 768 obserwacji, 9 kolumn  
**Populacja:** Kobiety z grupy etnicznej Pima Indian, wiek minimum 21 lat

| Kolumna | Opis |
|---|---|
| Pregnancies | Liczba ciąż |
| Glucose | Stężenie glukozy w osoczu (mg/dL) |
| BloodPressure | Rozkurczowe ciśnienie krwi (mm Hg) |
| SkinThickness | Grubość fałdu skórnego tricepsa (mm) |
| Insulin | Stężenie insuliny w surowicy (uU/mL) |
| BMI | Wskaźnik masy ciała (kg/m2) |
| DiabetesPedigreeFunction | Genetyczne ryzyko cukrzycy |
| Age | Wiek w latach |
| Outcome | Zmienna docelowa: 1 = cukrzyca, 0 = brak cukrzycy |

## Cel analizy

Projekt rozwiązuje problem klasyfikacji binarnej — na podstawie pomiarów
medycznych model przewiduje, czy u danej pacjentki występuje cukrzyca.

Główną metryką oceny jest recall dla klasy pozytywnej (cukrzyca), ponieważ
w zastosowaniach medycznych przeoczenie chorego pacjenta niesie wyższe
konsekwencje niż fałszywy alarm. Dodatkowo raportowane są: accuracy,
precision, F1-score oraz ROC-AUC.

## Struktura projektu

```
.
├── README.md
└── projekt_cukrzyca.ipynb
```

Notebook zawiera:
- eksploracyjną analizę danych (EDA)
- analizę korelacji między cechami
- czyszczenie danych i obsługę brakujących wartości
- budowę pipeline'u ML z użyciem scikit-learn
- porównanie trzech modeli klasyfikacji
- strojenie hiperparametrów metodą GridSearchCV
- ocenę finalnego modelu na zbiorze testowym
- wizualizacje: macierz konfuzji, krzywa ROC, ważność cech
- przykładowe predykcje dla nowych danych

## Metodologia

Dane zawierają ukryte brakujące wartości — biologicznie niemożliwe zera
w kolumnach takich jak Glucose czy BMI. Zastąpiono je medianą kolumny
przy użyciu SimpleImputer, który jest częścią Pipeline i działa wyłącznie
na danych treningowych, co eliminuje ryzyko data leakage.

Podział danych: 80% trening / 20% test z parametrem stratify, który
gwarantuje zachowanie proporcji klas w obu zbiorach.

Porównanie modeli przeprowadzono przy użyciu StratifiedKFold (k=5)
z metryką ROC-AUC. Najlepszy model poddano strojeniu hiperparametrów
przez GridSearchCV, po czym oceniono go jednorazowo na zbiorze testowym.

## Wyniki

Porównanie modeli w walidacji krzyżowej:

| Model | ROC-AUC (CV) |
|---|---|
| Logistic Regression | 0.82 |
| Random Forest | 0.84 |
| SVM | 0.82 |

Najlepsze wyniki uzyskał Random Forest. Po strojeniu przez GridSearchCV
(max_depth=5, min_samples_split=5, n_estimators=50) model osiągnął
na zbiorze testowym:

| Metryka | Wartość |
|---|---|
| Accuracy | 0.75 |
| ROC-AUC | 0.82 |
| Precision (cukrzyca) | 0.68 |
| Recall (cukrzyca) | 0.52 |
| F1-score (cukrzyca) | 0.59 |

Najważniejsza cecha predykcyjna to Glucose (ważność 0.35), następnie
BMI (0.19) i Age (0.12), co jest zgodne z kliniczną wiedzą medyczną.

## Technologie

- Python 3
- Google Colab
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

## Uruchomienie

Notebook pobiera dane automatycznie z publicznego repozytorium GitHub,
więc nie wymaga ręcznego pobierania plików. Wystarczy otworzyć go
w Google Colab i uruchomić wszystkie komórki.

## Literatura

Geron A., *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*, O'Reilly, 2019.
