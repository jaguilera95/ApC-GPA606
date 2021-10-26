'''
    Clase auxiliar para el manejo de los datos
    
    Autores: Juan Aguilera Toro
             Juan Manuel Camara Diaz
             Raul Salinas Natal
             
    Fecha: 2021-10-26
'''

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

def split_x_y_scale(df: pd.DataFrame, y_target: str):
    """
    Funció que retorna les variables x i y del dataframe df
    en formato numpy y normaliza X

    Args:
        df (pd.DataFrame): Dataframe
        y_target (String): Columna objetivo

    Returns:
        (np.array, np.array): X y y
    """
    
    y = df[y_target].to_numpy().astype('int')
    X = df.drop([y_target], axis="columns").to_numpy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    return X_s, y

def show_polinomial(X: np.array, y: np.array, coefs: list, bias: float):
    """
    Funcion que grafica los datos y el polinomio de grado n
    utilizando matplotlib.

    Args:
        X (np.array): X
        y (np.array): y
        coefs (list): Pesos del modelo
        bias (float): Bias del modelo
    """
    plt.scatter(X, y, color='red')
    X_seq = np.linspace(X.min(),X.max(),300).reshape(-1,1)
    plt.plot(X_seq, np.polyval(coefs, X_seq) + bias, color='blue', linewidth=2)
    plt.show()

def mse(v1: np.array, v2: np.array) -> float:
    """
    Funcion que retorna el mse entre los vectores v1 y v2

    Args:
        v1 (np.array): v1
        v2 (np.array): v2

    Returns:
        float: La mse entre v1 y v2
    """
    
    return ((v1 - v2)**2).mean()

def regression(x: np.array, y: np.array) -> LinearRegression:
    """
    Crea una regresión lineal y la entrena con los datos x y y

    Args:
        x (np.array): x
        y (np.array): y

    Returns:
        LinearRegression: Regresión lineal del paquete sklearn
    """
    
    regr = LinearRegression()
    regr.fit(x, y)
    return regr

def standarize(x_train: np.array) -> np.array:
    """
    Standariza los datos de x_train usando la media y desviación estándar

    Args:
        x_train (np.array): Datos a standarizar

    Returns:
        np.array: Datos standarizados
    """
    
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

def split_data(x: np.array, y: np.array, train_ratio=0.8):
    """
    Splitea los datos en train y test, usando el ratio train_ratio (por defecto 0.8) y de manera aleatoria.

    Args:
        x (np.array): X
        y ([type]): y
        train_ratio (float, optional): Porcentaje que de datos de entrenamiento. Defaults to 0.8.

    Returns:
        tuple(np.array): X_train, X_test, y_train, y_test
    """
    
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_test = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_test = x[indices_test, :]
    y_test = y[indices_test]
    return x_train, y_train, x_test, y_test

def split_smokers(df, target_column, normalize=True, correlation_min = None):
    """
    Funcion que separa el dataset en dos, uno con los fumadores y otro con los no fumadores.

    Args:
        df (pd.DataFrame): Dataset completo
        target_column (str): Columna objetivo
        normalize (bool, optional): Si es true, normaliza los datos. Defaults to True.
        correlation_min (float, optional): Valor minimo de coorelacion con la columna ojetivo. Defaults to None.

    Returns:
        Los datos separados por fumadores y no fumadores.
        Y listas de los nombres de las columnas de los datos separados por fumadores y no fumadores.
    """
    
    if correlation_min != None:
        corr = df.corr()
        corr_target = corr[target_column].abs().sort_values(ascending=False)
        corr_target = df[corr_target[corr_target < correlation_min].index].columns
        df_c = df.drop(columns=corr_target)
    else:
        df_c = df
        
    df_yes = df_c[df['smoker'] == 1]
    df_no  = df_c[df['smoker'] == 0]
    
    df_no = df_no.drop(columns=['smoker'])
    df_yes = df_yes.drop(columns=['smoker'])
    
    y_yes = df_yes[target_column].to_numpy().astype('int')
    X_yes = df_yes.drop([target_column], axis="columns").to_numpy()
    
    y_no  = df_no[target_column].to_numpy().astype('int')
    X_no  = df_no.drop([target_column], axis="columns").to_numpy()
    
    if normalize:
        scaler = StandardScaler()
        X_yes = scaler.fit_transform(X_yes)
        X_no = scaler.fit_transform(X_no)
    
    return X_no, y_no, X_yes, y_yes, df_no.columns, df_yes.columns

def caregorical_to_onehot_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Funcion que convierte una columna categórica en una columna de onehot

    Args:
        df (pd.DataFrame): Dataset
        col (str): Columna categórica

    Returns:
        pd.DataFrame: Dataset con la columna onehot
    """
    
    dummy = pd.get_dummies(df[col]) 
    df = df.drop(col, axis = 1) 
    return pd.concat([df, dummy], axis = 1)

def object_to_number(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """
    Funcion que convierte una columna de object a number.

    Args:
        df (pd.DataFrame): Dataset
        column (str): Columna de objetivo
        value (str): Valor a convertir

    Returns:
        pd.DataFrame: Dataset con la columna number
    """
    
    df_tmp = df[column].apply(lambda x: 1 if x == value else 0)
    df = df.drop(column, axis = 1) 
    return pd.concat([df, df_tmp], axis = 1)

def read_database(dir : str) -> pd.DataFrame:
    """
    Funcion que lee un archivo csv y lo convierte en un dataframe.

    Args:
        dir (str): Ruta del archivo csv

    Returns:
        pd.DataFrame: Dataset
    """
    
    return pd.read_csv(dir, delimiter= ',')

def show_data(df : pd.DataFrame) -> None:
    """
        Funcion que muestra los datos de un dataframe.
    """
    
    print(df.head(5))
    print(df.describe())
    print(df.info())

def barplot_gen(df_colum : pd.Series, ax = plt.subplot()):
    """
    Funcion que grafica una barra de una columna de un dataframe.

    Args:
        df_colum (pd.Series): Columna de un dataframe
        ax (plt.Subplot, optional): Sub-grafica. Defaults to plt.subplot().
    """
    
    classes = df_colum.value_counts()
    class_len = len(classes)
    sns.barplot(x=np.arange(class_len), y=classes, ax = ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes.index.values.tolist(), rotation=90, fontsize=15)
    ax.set_title(df_colum.name, fontsize=18)

def hist_gen(ax, df_colum : pd.Series):
    """
    Funcion que grafica un histograma de una columna de un dataframe.

    Args:
        ax (plt.Subplot): Sub-grafica
        df_colum (pd.Series): Columna de un dataframe
    """
    
    ax.set_title(f"Histograma de l'atribut {df_colum.name}")
    ax.set_xlabel("Attribute Value")
    ax.set_ylabel("Count")
    ax.hist(df_colum, bins=11, range=[np.min(df_colum), np.max(df_colum)], histtype="bar", rwidth=0.8)

def show_3d_plot(df: pd.DataFrame, column: pd.Series):
    """
    Funcion que muestra un grafico 3D de una columna de un dataframe.
    Despues de realizar una reduccion de dimensionalidad, se grafica el grafico 3D.

    Args:
        df (pd.DataFrame): Dataset
        column (pd.Series): Columna de un dataframe
    """
    
    y = df[column].to_numpy()
    X = df.drop([column], axis="columns").to_numpy()
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    pca=PCA(n_components=3)
    X_train_3dim = pca.fit_transform(X_s)

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(X_train_3dim[:,0], X_train_3dim[:,1], X_train_3dim[:,2], c=y)

    plt.show()

def show_2d_plot(df: pd.DataFrame, column: pd.Series):
    """
    Funcion que muestra un grafico 2D de una columna de un dataframe.
    Despues de realizar una reduccion de dimensionalidad, se grafica el grafico 2D.

    Args:
        df (pd.DataFrame): Dataset
        column (pd.Series): Columna de un dataframe
    """
    
    y = df[column].to_numpy()
    X = df.drop([column], axis="columns").to_numpy()
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    pca=PCA(n_components=2)
    X_train_3dim = pca.fit_transform(X_s)

    plt.scatter(X_train_3dim[:,0],X_train_3dim[:,1], c=y)
    plt.show()

def show_x_y(x: np.array, y: np.array, title: str, x_label: str, y_label: str):
    """
    Funcion que muestra un grafico de dos columnas de un dataframe.

    Args:
        x (np.array): Columna x
        y (np.array): Columna y
        title (str): Titulo del grafico
        x_label (str): Etiqueta x
        y_label (str): Etiqueta y
    """
    
    plt.xlim((min(x), max(x)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.scatter(x, y)
    plt.show()

def calculate_r2_score_atr(X: np.array, y: np.array, columns: list):
    """
    Funcion que calcula el R2 de una lista de atributos.

    Args:
        X (np.array): Dataset X
        y (np.array): Columna de objetivo
        columns (list): Nombre de las columnas
    """
    
    x_train, y_train, x_test, y_test = split_data(X, y)
    stndScal = StandardScaler()
    x_train = stndScal.fit_transform(x_train)
    x_test = stndScal.transform(x_test)
    error_np = np.zeros(x_train.shape[1])

    for i in range(x_train.shape[1]):
        x_t = x_train[:,i] # seleccionem atribut i en conjunt de train
        x_v = x_test[:,i] # seleccionem atribut i en conjunt de val.
        x_t = np.reshape(x_t,(x_t.shape[0],1))
        x_v = np.reshape(x_v,(x_v.shape[0],1))

        regr = regression(x_t, y_train)    
        error = mse(y_test, regr.predict(x_v)) # calculem error
        error_np[i] = error
        r2 = r2_score(y_test, regr.predict(x_v))

        print("Error en atribut %i %s: %f" %(i, columns[i], error))
        print("R2 score en atribut %i %s: %f" %(i, columns[i], r2))

def error_r2(X: np.array, y: np.array):
    """
    Funcion que calcula el error y el R2 de un conjunto de datos.

    Args:
        X (np.array): Dataset X
        y (np.array): Columna de objetivo
    """
    
    x_train, y_train, x_test, y_test = split_data(X, y)
    print(x_train.shape)

    regr = regression(x_train, y_train)
    error = mse(y_test, regr.predict(x_test))
    r2 = r2_score(y_test, regr.predict(x_test))

    print("Error: ", error, " R2: ", r2)
    
def calculate_best_dimension_pca(X: np.array, y: np.array) -> list:
    """
    Funcion que calcula la mejor dimension de un conjunto de datos.
    Utiliando el metodo PCA de la libreria sklearn.

    Args:
        X (np.array): Dataset X
        y (np.array): Columna de objetivo

    Returns:
        list(tuple): Lista con la dimension y el error
    """
    print(f'Original Dimension: {X.shape[1]}')
    x_train, y_train, x_test, y_test = split_data(X, y)
    results = []
    for d in range(1, x_train.shape[1] + 1):
        pca=PCA(n_components=d)
        x_train_d = pca.fit_transform(x_train)
        x_test_d = pca.transform(x_test)
        reg = regression(x_train_d, y_train)
        results.append((d, r2_score(reg.predict(x_test_d), y_test)))
        print("Dimension: ", d, " R2: ", r2_score(reg.predict(x_test_d), y_test))
    return results