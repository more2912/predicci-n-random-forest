import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Cargar el dataset desde el archivo CSV
data = pd.read_csv('csv/archive/supermarket_sales - Sheet1.csv', sep=';')

# Mostrar las primeras 5 filas del dataset
print(data.head())

# Resumen de las columnas y su tipo de dato
print(data.info())

# Resumen estadístico de las columnas numéricas
print(data.describe())


# Definir las variables independientes (X) y la variable dependiente (y)
X = data[['Unit price', 'Quantity']]  # Variables predictoras
y = data['Total']  # Variable objetivo

# Dividir los datos en conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las dimensiones de los conjuntos de entrenamiento y prueba
print(f'Conjunto de entrenamiento: {X_train.shape[0]} muestras')
print(f'Conjunto de prueba: {X_test.shape[0]} muestras')

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Mostrar las primeras predicciones y los valores reales para comparar
print("Predicciones:", y_pred[:5])#muestra las primeras 5 predicciones
print("Valores Reales:", y_test.values[:5])

# Calcular el error absoluto medio (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Error Absoluto Medio (MAE): {mae}')

# Comparación de ventas reales vs predichas
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Comparación de Ventas Reales y Predichas')
plt.show()