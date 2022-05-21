# Schneider Electric Hackaton!

Juan Sirai

-----------------------------------------------------------------------

El presente trabajo se encuentra enmarcado dentro del Hackaton Scheider Electric, y se resumen en los principales apartados:

`Preprocesamiento`

Se obtienen 6 datasets distintos para entrenar el modelo, de las siguientes fuentes:
* CSV: se trabajan directamente con pandas
* Json: se consulta directamente en endpoint mediante la libreria requests
* PDF: se utiliza PyPDF2 para abrir y trabajar el archivo

`Análisis Exploratorio de variables`

A fin de tener un mejor conocimiento de la interacción de las distintas variables entre si, y con nuestro target, utilizamos la libreria seaborn y matplotlib a fin de plotear las principales relaciones.

`Modelado y Evaluación`

Dado el tiempo acotado para este proyecto, se probó solamente con dos algoritmos que me han resultado confiables en situaciones similares:

Random Forest: En Random Forest se ejecutan varios algoritmos de árbol de decisiones en lugar de uno solo. Para clasificar un nuevo objeto basado en atributos,
cada árbol de decisión da una clasificación y finalmente la decisión con mayor “votos” es la predicción del algoritmo.

Stochastic Gradient Decent Classifier: algoritmo relativamente nuevo para mi, en este ejercicio me resultó algo inexacto, probablemente por falta de experiencia en su optimizacion.

La evaluación de ambos modelos se realizó considerando el F1 score, resultando Random Forest el mas favorable

`Resultados`

Se guardan tanto en formato CSV como Json.

## Librerias necesarias:

Para ejecutar el presente proyecto, se necesitarán las importaciones de Python:

```python
# Para procesado
import pandas as pd
import requests
import json
import PyPDF2
import re
import numpy as np
from os import listdir
from os.path import isfile, join
import glob

# Librerias para graficar
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## Modelado
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
```
