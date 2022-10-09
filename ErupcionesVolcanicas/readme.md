# Hackathon Jornada de Talent Digital

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Mayon_volcano_banner.jpg/1200px-Mayon_volcano_banner.jpg">

## Clasificacion de Erupciones Volcanicas
#### Sirai Juan

---------------------------------------------------------------------
# Abstracto

Jorge es un geólogo del IGME (Instituto Geológico y Minero de España) que está desarrollando un nuevo sistema de prevención 
de erupciones para poder predecir qué tipo de erupción tendrá un volcán según las las vibraciones detectadas por 
sus sensores durante los días previos a la erupción. Esto permitirá reducir el riesgo de víctimas y destrozos materiales por este tipo de catástrofe natural.
El sistema de Jorge trabaja con 5 tipos de erupciones:

<img src="https://challenges-asset-files.s3.us-east-2.amazonaws.com/data_sets/Data-Science/4+-+events/jobmadrid/images/tipos.jpeg" width=50%>

* **Pliniana**: Se caracteriza por su alto grado de explosividad, con manifestaciones muy violentas en las cuales se expulsan grandes 
* volúmenes de gas volcánico, fragmentos y cenizas.
* **Peleana**: La característica más importante de una erupción peleana es la presencia de una avalancha brillante de ceniza volcánica caliente, 
* llamada flujo piroclástico.
* **Vulcaniana**: Son erupciones volcánicas de tipo explosivo. El material magmático liberado es más viscoso que en el caso de las erupciones 
* hawaianas o estrombolianas; consecuentemente, se acumula más presión desde la cámara magmática conforme el magma asciende hacia la superficie.
* **Hawaiana**: Consiste en la emisión de material volcánico, mayoritariamente basáltico, de manera efusiva o no explosiva. Ocurre de este modo 
* debido a que la difusión de los gases a través de magmas más básicos (basálticos) puede hacerse de manera lenta pero más o menos continua.
*  Consecuentemente, las erupciones volcánicas de este tipo no suelen ser muy destructivas.
* **Estromboliana**: La erupción Estromboliana está caracterizada por erupciones explosivas separadas por periodos de calma de duración variable.
*  El proceso de cada explosión corresponde a la evolución de una burbuja de gases liberados por el propio magma.

El objetivo de este reto será ayudar a Jorge realizando el modelado predictivo a partir de un dataset que contiene las mediciones hechas por sus sensores y tipos.

# Dataset

**Variables del dataset:**

<u>Features:</u> El dataset contiene 6 features en 6 columnas, que son los parámetros medidos por los diferentes sensores. Estos corresponden 
a las vibraciones detectadas en ciertos puntos de la ladera del volcán.

<u>Target:</u> El target corresponde al 'label' que clasifica los tipos de erupciones volcánicas en función de los features medidos por los sensores.

* Target 0 corresponde a una erupción de tipo Pliniana
* Target 1 corresponde a una erupción de tipo Peleana
* Target 2 corresponde a una erupción de tipo Vulcaniana
* Target 3 corresponde a una erupción de tipo Hawaiana
* Target 4 corresponde a una erupción de tipo Estromboliana

**Archivos:**

jm_train.csv: Este dataset contiene tanto las variables predictoras como el tipo de erupción.

jm_test_X.csv: Este dataset contiene las variables predictoras con las que se tendrá que predecir el tipo de erupción.

---------------------------------------------------------------

# Librerias

```python
import os
# Para EDA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Para Modelado
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
```

--------------------------------------------------------------------------
# Analisis Exploratorio (EDA)

En este apartado, se realizaron algunos análisis básicos del dataset de entrenamiento, a fin de determinar el estado del mismo en cuanto a 
completitud de datos, escalas, tipo de variables, así como también descubrir cómo se distribuyen las features predictoras, y su relación con el target.

Algunas conclusiones preliminares:

**Calidad de datos**

El dataset se encuentra completo, sin valores null ni blanks.<br>
Son mayoritariamente variables cuantitativas, a excepción del target que para fines de análisis debería tratarse como categórica 
(aunque luego nos sirve que esté como integer para fines de modelado)

|count	|mean|	std|	min|	25%|	50%|	75%|	max|
|--|--|--|--|--|--|--|--|
|feature1	|2100.0|	-0.204656	|1.543613	|-6.683655|	-1.171340	|-0.443868	|0.890023	|4.565547|
|feature2	|2100.0	|0.199249	|1.614024	|-5.383371	|-0.877386	|0.320507	|1.321430	|6.700133|
|feature3	|2100.0	|-0.378140	|1.450548	|-6.147055|	-1.365990	|-0.439745	|0.662898|	4.157518|
|feature4	|2100.0	|-0.206425	|1.442225	|-5.653594	|-1.259403	|-0.176504	|0.866879|	4.371912|
|feature5	|2100.0	|-0.186419	|1.501573	|-5.912521	|-1.211685	|-0.195751	|0.851843	|5.068783|
|feature6	|2100.0	|-0.433229	|1.188791	|-4.433189	|-1.131944	|-0.406754	|0.348593	|4.016324|
|target	|2100.0	|1.980476	|1.410537	|0.000000	|1.000000	|2.000000|	3.000000	|4.000000|

La distribución de variables se centra en una media cercana a 0, y un desvío standard que ronda los 1.5
Si bien las escalas no son disímiles, no tenemos mayor información para asumir que los sensores miden en la misma unidad.

**Correlacion entre features**

* La variable target tiene correlación fuerte y en sentido positivo con "Feature 1" y "Feature 5"
* Las variables "Feature 5" y "Feature 3", así como "Feature 3" y "Feature 6" tienen fuerte correlación entre si. Esto podría ocasionar problemas si implementamos a futuro modelos lineales.

--------------------------------------------------------
# Modelos

A lo largo del desafío, se entrenaron e implementaron los siguientes modelos:

**Random Forest Classifier**:<br>
En Random Forest se ejecutan varios algoritmos de árbol de decisiones en lugar de uno solo. 
Para clasificar un nuevo objeto basado en atributos, cada árbol de decisión da una clasificación y finalmente la decisión con mayor
“votos” es la predicción del algoritmo.

**Stochastic Gradient Decent Classifier**<br>
La clase SGDClassifier implementa una rutina de aprendizaje de descenso de gradiente estocástico simple 
que soporta diferentes funciones de pérdida y penalizaciones para la clasificación.
Al igual que otros clasificadores, el SGD debe estar equipado con dos matrices: una matriz X de tamaño
[n_muestras, n_funciones] que contenga las muestras de formación, y una matriz Y de tamaño[n_muestras] 
que contenga los valores objetivo (etiquetas de clase) para las muestras de formación.

**Support Vector Classifier**<br>
Las Máquinas de Vectores de Soporte (Support Vector Machines) permiten encontrar la forma óptima de clasificar entre varias clases.
La clasificación óptima se realiza maximizando el margen de separación entre las clases. Los vectores que definen el borde de esta separación son 
los vectores de soporte.

**Gradient Boost Classifier**<br>
Gradient boosting o Potenciación del gradiente, es una técnica de aprendizaje automático utilizado para el análisis de la regresión
y para problemas de clasificación estadística, el cual produce un modelo predictivo en forma de un conjunto de modelos de predicción débiles, 
típicamente árboles de decisión.

**ADA Boost Classifier**<br>
AdaBoost es una contraccion de “Adaptive Boosting”, en donde el termino Adaptive
hace alusion a su principal diferencia con su predecesor. En terminos de funcionalidad son
iguales, ambos algoritmos buscan crear un clasificador fuerte cuya base sea la combinaci´on
lineal de clasificadores “debiles simples” ht(x). Sin embargo, AdaBoost propone entrenar
una serie de clasificadores debiles de manera iterativa, de modo que cada nuevo clasificador
o “weak learner” se enfoque en los datos que fueron erroneamente clasificados por su
predecesor, de esta manera el algoritmo se adapta y logra obtener mejores resultados.

--------------------------------------------------------

# Seleccion y Prediccion

Para la predicción final, se eligió el modelo SVC optimizado con GridSearchCV, y con features estandarizadas.

Las métricas finales fueron:

```

              precision    recall  f1-score   support

           0       0.83      0.77      0.79        81
           1       0.76      0.74      0.75        91
           2       0.78      0.72      0.75        88
           3       0.71      0.87      0.78        76
           4       0.82      0.81      0.81        84

    accuracy                           0.78       420
   macro avg       0.78      0.78      0.78       420
weighted avg       0.78      0.78      0.78       420

```

------------------------------------------------------

# Aclaraciones

Durante el entrenamiento de los distintos modelos, se probó una versión del dataset que no incluía la feature 3 (ya que poseía un grado ligeramente fuerte de 
correlación con otras dos features y podría romper los principios de normalidad para los modelos lineales).<br>
No obstante, la performance de dicho modelo decreció, con lo cual para la versión final se opta por incluirla.
