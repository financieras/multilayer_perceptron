# MULTILAYER PERCEPTRON

## INSTALACIÓN

```bash
# Clonar el repositorio del proyecto 'perceptron'

git clone https://github.com/financieras/perceptron.git


# Ir a la carpeta del proyecto

cd perceptron


# Crear un entorno virtual llamado '.venv'
# El nombre lleva punto para que sea oculto y no moleste visualmente
# Debes estar en la carpeta del proyecto

python3 -m venv .venv


# Activar el entorno virtual

source .venv/bin/activate
# Ahora verás que al inicio del promt pone (.venv)

# Recomendable: actualiza PIP

pip install --upgrade pip


# Instalar los paquetes necesarios para el proyecto dentro del entorno virtual
# Instalar las dependencias listadas en el archivo requirements.txt en tu entorno virtual

pip install -r requirements.txt
 

# Entrar a Jupyter Lab

jupyter lab


# Optativo: al terminar de trabajar, desactivar el entorno virtual con el comando:

deactivate

```


## FASES

1. PREPROCESADO DE LOS DATOS

- Observamos que hay 32 registros sin cabecera.
- Vemos que no hay filas vacías ni datos repetidos
- Creamos la cabecera:
  - Primera columna es el ID
  - Segunda columna es el diagnóstico (`diagnosis`) que toma el valor M (Maligno) o B (Benigno). Es la variable a estimar.
  - Treinta columna numéricas de tipo float que nombramos como frature01, ..., feature30.
- Normalizamos las 30 columnas usando la media y la desviación estandar
- Convertimos la columna `diagnosis` en un binario: 1 (M) y 0 (B)

1. DIVISION DE LOS DATOS EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA

- Dividimos los 569 registros en un conjunto de entrenamiento (420 registros) y un conjunto de prueba (149 registros).
- Podemos utilizar una función como train_test_split() de la librería sklearn.model_selection para realizar esta división de manera aleatoria.

3. DISEÑO DE LA ARQUITECTURA DE LA RED NEURONAL

- Capa de entrada: 30 neuronas, una por cada característica numérica.
- 2 capas ocultas:
    1. La primera capa oculta con 16 neuronas
    2. La segunda capa oculta con 8 neuronas
- Capa de salida: 1 neurona, que represente la predicción de diagnóstico (1 para M (Maligno), 0 para B (Benigno)).
- Resumiendo: las capas cuatro del modelo contienen este número de neuronas (30 → 16 → 8 → 1)

- Es común utilizar funciones de activación como **ReLU** (Rectified Linear Unit) en las capas ocultas y una función **sigmoide** en la capa de salida para obtener valores entre 0 y 1.
  Luego esos valores se convierten en 0 si son menores de 0.5 y se convierten en 1 si son mayores o iguales.

4. ENTRENAMIENTO DEL MODELO

- Definir la función de pérdida (por ejemplo, entropía cruzada binaria) y el optimizador (por ejemplo, Stochastic Gradient Descent).
- Entrenar el modelo utilizando el conjunto de entrenamiento, monitorizando el rendimiento en el conjunto de validación.
- Ajustar hiperparámetros como la tasa de aprendizaje, el número de épocas, el tamaño del lote, etc. para mejorar el rendimiento.

5. EVALUACIÓN DEL MODELO

- Evaluar el rendimiento del modelo entrenado utilizando el conjunto de prueba.
- Calcular métricas como exactitud, precisión, exhaustividad y puntuación F1.
- Analizar el rendimiento del modelo y determinar si es adecuado para el problema.

6. AJUSTE Y OPTIMIZACIÓN DEL MODELO

- Si el rendimiento no es satisfactorio, pueden probarse diferentes arquitecturas de red, modificar los hiperparámetros o explorar técnicas de regularización.
- En un caso real también se puede considerar la incorporación de técnicas de aumento de datos o ingeniería de características adicionales.

## MATEMÁTICAS

# Forward Propagation en Perceptrón Multicapa

## Notación Básica
- $x_i$: Valor de entrada i-ésimo
- $w_{jk}^{[l]}$: Peso de la conexión entre la neurona k de la capa l-1 y la neurona j de la capa l
- $b_j^{[l]}$: Sesgo (bias) de la neurona j en la capa l
- $z_j^{[l]}$: Entrada ponderada de la neurona j en la capa l
- $a_j^{[l]}$: Activación (salida) de la neurona j en la capa l
- $\sigma()$: Función de activación

## Ecuaciones Fundamentales

### 1. Entrada Ponderada
Para cada neurona j en la capa l:

$$z_j^{[l]} = \sum_{k=1}^{n^{[l-1]}} w_{jk}^{[l]}a_k^{[l-1]} + b_j^{[l]}$$

donde $n^{[l-1]}$ es el número de neuronas en la capa anterior.

### 2. Activación
La salida de cada neurona se obtiene aplicando la función de activación:

$$a_j^{[l]} = \sigma(z_j^{[l]})$$

### 3. Funciones de Activación Comunes

#### ReLU (para capas ocultas)
$$\sigma(z) = \max(0,z)$$

#### Sigmoide (para la capa de salida en clasificación binaria)
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

## Notación Matricial

Para una capa l completa:

$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = \sigma(Z^{[l]})$$

Donde:
- $W^{[l]}$$: Matriz de pesos de la capa l
- $A^{[l-1]}$: Vector de activaciones de la capa anterior
- $b^{[l]}$: Vector de sesgos de la capa l
- $Z^{[l]}$: Vector de entradas ponderadas
- $A^{[l]}$: Vector de activaciones de la capa l

## Dimensiones de las Matrices

Para una red con capas de tamaño $n^{[l-1]}$ y $n^{[l]}$:
- $W^{[l]}$: Matriz de dimensión $(n^{[l]}, n^{[l-1]})$
- $A^{[l-1]}$: Vector de dimensión $(n^{[l-1]}, 1)$
- $b^{[l]}$: Vector de dimensión $(n^{[l]}, 1)$
- $Z^{[l]}$, $A^{[l]}$: Vectores de dimensión $(n^{[l]}, 1)$

## Ejemplo para tu Red Específica

Para tu red con 30 entradas, capas ocultas de 10 neuronas y 1 salida:

### Primera Capa Oculta
- $W^{[1]}$: Matriz de $(10, 30)$
- $A^{[0]}$ (entradas): Vector de $(30, 1)$
- $b^{[1]}$: Vector de $(10, 1)$
- $Z^{[1]}$, $A^{[1]}$: Vectores de $(10, 1)$

### Capa de Salida
- $W^{[final]}$: Matriz de $(1, 10)$
- $b^{[final]}$: Vector de $(1, 1)$
- $Z^{[final]}$, $A^{[final]}$: Escalares (vectores de $(1, 1)$)