import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Cargar los datos con el delimitador correcto
data = pd.read_csv('csv/archive/supermarket_sales - Sheet1.csv', sep=';')

# Convertir las variables categóricas en variables dummy (One-Hot Encoding)
data_encoded = pd.get_dummies(data, columns=['Product line', 'Customer type', 'Gender', 'Payment'], drop_first=True)

# Verificar las nuevas columnas generadas después del One-Hot Encoding
print(data_encoded.columns)

# Ajustar la selección de columnas en función de los nombres generados
X = data_encoded[['Unit price', 'Quantity',
                  'Product line_Fashion accessories', 
                  'Product line_Food and beverages',
                  'Product line_Health and beauty',
                  'Product line_Home and lifestyle', 
                  'Product line_Sports and travel',
                  'Customer type_Normal', 
                  'Gender_Male', 
                  'Payment_Credit card', 
                  'Payment_Ewallet']]

# Variable objetivo
y = data_encoded['Total']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Random Forest
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)

# Entrenar el modelo con los datos de entrenamiento
rf_model.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred_rf = rf_model.predict(X_test)

# Evaluar el modelo de Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print(f'Error Absoluto Medio (MAE) con Random Forest: {mae_rf}')

#
