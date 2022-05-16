# JOBarcelona ’22 | Data Science
*Este es uno de los retos clasificatorios que forman parte del hackathon online de JOBarcelona ’22. 
El resultado de este reto va a permitir a los ganadores asistir al hackathon presencial que se realizará el día 31 de mayo de 2022 en el Camp Nou.*

<img src="https://challenges-asset-files.s3.us-east-2.amazonaws.com/JOBarcelona+2022/data.png" width=800>

## Introducción

Los insectos nocturnos representan uno de los grupos más diversos de organismos, por lo que es de suma importancia estudiarlos.

Es por ello que un grupo de prestigiosos entomólogos han construido un ecosistema aislado con múltiples especies para poder estudiarlos en mayor detalle. 
Para este estudio están diseñando un sistema de sensores para poder trackear de forma automática las dinámicas y hábitos de estos insectos.


## Entrega

* El código del modelo predictivo.
* Archivo results.csv con las predicciones del modelo

## Usage

Librerías necesarias:
Para poder correr localmente el algoritmo, deberá realizar los siguientes import
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
```

## Secciones

El presente trabajo, se divide en las siguientes secciones:
* **Análisis preliminar y descriptivo:** en donde se realiza la importación de los datos, se analiza la calidad de los mismos y se busca entender la relación de las variables entre sí y con la variable target
* **Transformaciones:** Se realizan todas las transformaciones necesarias sobre el set de entrenamiento a fin de poder fitear el modelo
* **Analisis de modelos:** A partir del set de entrenamiento, se analiza la performance de distintos modelos de clasificación
* **Predicción:** Se realiza la predicción final sobre datos completamente nuevos, y se almacena la salida en un file .csv

## License
n/a
