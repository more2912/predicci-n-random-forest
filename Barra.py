import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Cargar el dataset desde el archivo CSV
data = pd.read_csv('csv/archive/supermarket_sales - Sheet1.csv',sep=';')

# Mostrar las primeras 5 filas del dataset
print(data.head())

# Resumen de las columnas y su tipo de dato
print(data.info())

# Resumen estadístico de las columnas numéricas
print(data.describe())

# Gráfico de barras de las ventas totales por línea de producto
sns.barplot(x='Product line', y='Total', data=data, estimator=sum, ci=None, palette='muted')#crea gráfico de barras
plt.title('Ventas Totales por Línea de Producto')
plt.xticks(rotation=45)
plt.ylabel('Total de Ventas')
plt.show()

# Gráfico de barras para ventas por género
sns.barplot(x='Gender', y='Total', data=data, estimator=sum, ci=None, palette='muted')
plt.title('Ventas por Género')
plt.ylabel('Total de Ventas')
plt.show()

