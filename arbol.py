import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Cargar los datos con el delimitador correcto
data = pd.read_csv('csv/archive/supermarket_sales - Sheet1.csv', sep=';')

# Verificar las columnas para asegurarnos de que se han cargado correctamente
print(data.columns)

# Convertir las variables categóricas en variables dummy (One-Hot Encoding)
data_encoded = pd.get_dummies(data, columns=['Product line', 'Customer type', 'Gender', 'Payment'], drop_first=True)

# Definir las variables predictoras (incluyendo las nuevas variables categóricas)
X = data_encoded[['Unit price', 'Quantity',
                  'Product line_Fashion accessories', 'Product line_Food and beverages',
                  'Product line_Health and beauty', 'Product line_Home and lifestyle', 
                  'Gender_Male', 'Payment_Ewallet']]

# Variable objetivo
y = data_encoded['Total']


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
print(f'Error Absoluto Medio (MAE) después de agregar variables: {mae}')

# Visualización de los resultados
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Comparación de Ventas Reales y Predichas (con más variables)')
plt.show()

# Crear el modelo de Árbol de Decisión
tree_model = DecisionTreeRegressor(random_state=42)

# Entrenar el modelo con los datos de entrenamiento
tree_model.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred_tree = tree_model.predict(X_test)

# Evaluar el modelo de Árbol de Decisión
mae_tree = mean_absolute_error(y_test, y_pred_tree)
print(f'Error Absoluto Medio (MAE) con Árbol de Decisión: {mae_tree}')

# Visualización de los resultados
plt.scatter(y_test, y_pred_tree, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Ventas Reales')
plt.ylabel('Ventas Predichas')
plt.title('Comparación de Ventas Reales y Predichas (Árbol de Decisión)')
plt.show()
