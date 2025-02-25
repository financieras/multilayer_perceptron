import pandas as pd

# Cargar los datos preprocesados
data = pd.read_csv('data/preprocessed_data.csv')

# Separar las características (X) y el objetivo (y)
X = data.iloc[:, 2:].values  # Características (features)
y = data['diagnosis'].values  # Objetivo (diagnóstico)

# Verificar las dimensiones de X
print(f"Dimensiones de X: {X.shape}")