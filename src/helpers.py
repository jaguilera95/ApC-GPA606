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

PLOT_DATA = False
DATA_DIR = 'data/insurance.csv'

def split_x_y_scale(df, y_target):
    y = df[y_target].to_numpy().astype('int')
    X = df.drop([y_target], axis="columns").to_numpy()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    return X_s, y

def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

def split_data(x, y, train_ratio=0.8):
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

def caregorical_to_onehot_encode(df, col):
    dummy = pd.get_dummies(df[col]) 
    df = df.drop(col, axis = 1) 
    return pd.concat([df, dummy], axis = 1)

def object_to_number(df, column, value):
    df_tmp = df[column].apply(lambda x: 1 if x == value else 0)
    df = df.drop(column, axis = 1) 
    return pd.concat([df, df_tmp], axis = 1)

def read_database(dir : str):
    return pd.read_csv(dir, delimiter= ',')

def show_data(df : pd.DataFrame):
    print(df.head(5))
    print(df.describe())
    print(df.info())

def barplot_gen(df_colum : pd.Series, ax = plt.subplot()):
    classes = df_colum.value_counts()
    class_len = len(classes)
    sns.barplot(x=np.arange(class_len), y=classes, ax = ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes.index.values.tolist(), rotation=90, fontsize=15)
    ax.set_title(df_colum.name, fontsize=18)

def hist_gen(ax, df_colum : pd.Series):
    ax.set_title(f"Histograma de l'atribut {df_colum.name}")
    ax.set_xlabel("Attribute Value")
    ax.set_ylabel("Count")
    ax.hist(df_colum, bins=11, range=[np.min(df_colum), np.max(df_colum)], histtype="bar", rwidth=0.8)

def show_3d_plot(df, column):
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

def show_2d_plot(df, column):
    y = df[column].to_numpy()
    X = df.drop([column], axis="columns").to_numpy()
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    pca=PCA(n_components=2)
    X_train_3dim = pca.fit_transform(X_s)

    plt.scatter(X_train_3dim[:,0],X_train_3dim[:,1], c=y)
    plt.show()

def show_x_y(x, y, title, x_label, y_label):
    plt.xlim((min(x), max(x)))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.scatter(x, y)
    plt.show()

def calculate_r2_score_atr(X, y, columns):
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

def error_r2(X, y):
    x_train, y_train, x_test, y_test = split_data(X, y)
    print(x_train.shape)

    regr = regression(x_train, y_train)
    error = mse(y_test, regr.predict(x_test))
    r2 = r2_score(y_test, regr.predict(x_test))

    print("Error: ", error, " R2: ", r2)
    
def calculate_best_dimension_pca(X, y):
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