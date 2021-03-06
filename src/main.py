'''
    Practica 1: Regresion Lineal
    
    Autores: Juan Aguilera Toro
             Juan Manuel Camara Diaz
             Raul Salinas Natal
             
    Fecha: 2021-10-26
'''

import seaborn as sns
import matplotlib.pyplot as plt
from helpers import *
from regression import Regression

###################
#    IMPORTANTE   #
###################

# Recomendamos ejecutar el codigo poco a poco por cada comentario que hay en el codigo.

PLOT_DATA = True # False para no mostrar los graficos de los apartados
DATA_DIR = 'data/insurance.csv'

if not PLOT_DATA:
    plt.show = lambda: None

def run():
    df = read_database(DATA_DIR)
    
    df = apartado_c(df)
    apartado_b(df)
    apartado_a(df)

def apartado_c(df):    
    print('''
    # --------------------------------------------------
    # Apartado C                                       #
    # --------------------------------------------------
    ''')
    
    # Mostramos las 5 primeras filas, la descripcion y la info de la base de datos.
    show_data(df)
    
    # Mostramos el grafico de barras de las clases de la columna 'smoker'
    barplot_gen(df['smoker'])
    plt.show()
    
    # Mostramos el grafico de barras de las clases de la columna 'region' y 'sex' 
    _, p = plt.subplots(ncols=2, figsize=(12, 3))
    barplot_gen(df['sex'], ax = p[0])
    barplot_gen(df['region'], ax = p[1])
    plt.show()
    
    _, p = plt.subplots(2, 2, sharey=True, figsize=(12, 10))

    # Mostramos histogramas de 'bmi', 'expenses', 'age' y 'children'
    hist_gen(p[0,0], df['bmi'])
    hist_gen(p[0,1], df['expenses'])
    hist_gen(p[1,0], df['age'])
    hist_gen(p[1,1], df['children'])
    plt.show()
    
    # Mostramos el pairplot de la base de datos de los valores numericos
    sns.pairplot(df)
    plt.show()
    
    # Pasamos los valores binarios (categoricos) a una variable numerica
    df = object_to_number(df, 'sex', 'male')
    df = object_to_number(df, 'smoker', 'yes')

    # Pasamos los valores categoricos a variables numericas
    df = caregorical_to_onehot_encode(df, 'region')

    # Mostramos la correlacion de las variables
    correlacio = df.corr()
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(correlacio, annot=True, vmin=-1, linewidths=.5, cmap=plt.cm.Blues)
    plt.show()
    
    # Agrupamos los datos por la columna 'smoker' para ver los gastos de los que fuman
    print(df.groupby("smoker").expenses.agg(["mean","median","count"]))
    
    # This function combines regplot() and FacetGrid.
    # It is intended as a convenient interface to fit regression
    # models across conditional subsets of a dataset.
    sns.lmplot(x="bmi", y="expenses", hue="smoker", data=df)
    plt.show()
    
    # Gives a better representation of the distribution of values,
    # but it does not scale well to large numbers of observations.
    # This style of plot is sometimes called a ???beeswarm???.
    ax = sns.swarmplot(x=df['smoker'], y=df['expenses'])
    ax.set_title("Smoker vs Expenses")
    plt.xlabel("Smoker (Yes - 1, No - 0)")
    plt.ylabel("Expenses")
    plt.show()
    
    # Realiza un reduccion de dimensionalidad de los datos y los muestra en una grafica
    show_2d_plot(df, 'smoker')
    
    return df

def apartado_b(df):
    print('''
    # --------------------------------------------------
    # Apartado B                                       #
    # --------------------------------------------------
     ''')
    
    # Creamos 2 datasets por cada clase de la columna 'smoker'
    # donde el valor objetivo es 'expenses'
    X_no, y_no, X_yes, y_yes, no_columns, yes_columns = split_smokers(df, 'expenses')
    
    # Visluaizamos los datos nuesvos separados por la clase smoker y sus regresiones.
    
    show_x_y(X_no[:,0], y_no, "Expenses vs. Age", "Age (Scaled)", "Expenses")
    dummy = X_no[:,0].reshape(X_no.shape[0],1)
    r = regression(dummy, y_no)
    plt.title("Expenses vs. Age No smoke")
    show_regresion(dummy, y_no, r)

    show_x_y(X_no[:,1], y_no, "Expenses vs. Bmi", "Bmi (Scaled)", "Expenses")
    dummy = X_no[:,1].reshape(X_no.shape[0],1)
    r = regression(dummy, y_no)
    plt.title("Expenses vs. Bmi No smoke")
    show_regresion(dummy, y_no, r)

    show_x_y(X_yes[:,1], y_yes, "Expenses vs. Bmi", "Bmi (Scaled)", "Expenses")
    dummy = X_yes[:,1].reshape(X_yes.shape[0],1)
    r = regression(dummy, y_yes)
    plt.title("Expenses vs. Bmi Smoke")
    show_regresion(dummy, y_yes, r)

    show_x_y(X_yes[:,0], y_yes, "Expenses vs. Age", "Age (Scaled)", "Expenses")
    dummy = X_yes[:,0].reshape(X_yes.shape[0],1)
    r = regression(dummy, y_yes)
    plt.title("Expenses vs. Age Smoke")
    show_regresion(dummy, y_yes, r)

    print('\n-----------------------\nYes r2 score\n-----------------------')
    calculate_r2_score_atr(X_yes, y_yes, yes_columns)

    print('\n-----------------------\nNo r2 score\n-----------------------')
    calculate_r2_score_atr(X_no, y_no, no_columns)
    
    # Comparacion de como mejora la regresion cuando descartamos los valores con un R2 score menos a 0.1
    
    print('No Filter')

    X_no, y_no, X_yes, y_yes, no_columns, yes_columns = split_smokers(df, 'expenses', True, None)

    print('Yes smoke r2 score')
    error_r2(X_no, y_no)
    print('No smoke r2 score')
    error_r2(X_yes, y_yes)

    print('Filter')

    X_no, y_no, X_yes, y_yes, no_columns, yes_columns = split_smokers(df, 'expenses', True, 0.1)

    print('Yes smoke r2 score')
    error_r2(X_no, y_no)
    print('No smoke r2 score')
    error_r2(X_yes, y_yes)
    
    # Calculamos la mejor dimension para el dataset
    
    X, y = split_x_y_scale(df, 'expenses')
    results = calculate_best_dimension_pca(X, y)
    print(f'La mejor dimension es {max(results, key = lambda x: x[1])}')
    plt.plot([x[0] for x in results], [x[1] for x in results])
    plt.show()
    
    # Calcular la mejor dimension posible con mle
    # https://tminka.github.io/papers/pca/minka-pca.pdf
    
    pca = PCA(n_components='mle', svd_solver='full')
    best_d = pca.fit_transform(X)
    
    print(f'La mejor dimension para X es: {best_d.shape[1]}')

def apartado_a(df):
    print('''
    # --------------------------------------------------
    # Apartado A                                       #
    # --------------------------------------------------
    ''')
    
    # Comparamos nuestro regresor con un modelo polinomial
    # para ver si lo hemos relizado correctamente
    # Hemos deccidido realiza las pruebas solo con los datos de los que fuman
    
    _, _, X_yes, y_yes, _, _ = split_smokers(df, 'expenses', True, 0.1)

    x_train_yes, y_train_yes, x_test_yes, y_test_yes = split_data(X_yes, y_yes)
    
    for x in range(1, 5):
        
        test_degree = x

        kw = {'lr': 0.01, 'l': 0.0001, 'epsilon': 0.1, 'degree': test_degree}

        dummy_yes = x_train_yes[:,1].reshape(x_train_yes.shape[0], 1)
        dummy_yes_test = x_test_yes[:,1].reshape(x_test_yes.shape[0], 1)

        r = train_degree(dummy_yes, y_train_yes, dummy_yes_test, y_test_yes, **kw)
        r_sk = train_degree_sk(dummy_yes, y_train_yes, dummy_yes_test, y_test_yes, test_degree)

        # Comparamos nuestra implmentacion con sklearn
        
        print(f'El valor de los pesos de nuestra implementacion es: {r[1]}')
        print(f'El valor de los pesos de la implementacion de sklearn es: {r_sk[1]}')
        print(f'El valor del bias de nuestra implementacion es de: {r[2]}')
        print(f'El valor del bias de la implementacion de sklearn es: {r_sk[2]}')

        print(f'El r2 score de nuestra implementacion es: {r[0]}')
        print(f'El r2 score de la implementacion de sklearn es: {r_sk[0]}')

        plt.title('Modelo de nuestra implementacion')
        show_polinomial(dummy_yes, y_train_yes, r[1], r[2])
        plt.title('Modelo de sklearn')
        show_polinomial(dummy_yes, y_train_yes, r_sk[1], r_sk[2])
    
    # Visualizacion del coeficiente de prismatico
    
    x_val = x_train_yes
    y_val = y_train_yes
    regr = regression(x_val, y_val)
    predX3D = regr.predict(x_val)
    
    coeficiente_prismatico(x_val, y_val, predX3D)
    plt.show()
    
    regr = Regression(x_val, y_val)
    regr.train()
    predX3D = regr.predict(x_val)
    coeficiente_prismatico(x_val, y_val, predX3D)
    plt.show()
    
    # Comparativa entre un modelo polinomial y un modelo de regresion lineal
    
    mse_n = []
    mse_p = []

    for _ in range(20):
        kw = {'lr': 0.01, 'l': 0.0001, 'epsilon': 0.1, 'degree': 2}
        r = train_degree(x_train_yes, y_train_yes, x_test_yes, y_test_yes, **kw)
        mse_n.append(r[-1][-1])
        
        kw = {'lr': 0.01, 'l': 0.0001, 'epsilon': 0.1, 'degree': 3}
        r2 = train_degree(x_train_yes, y_train_yes, x_test_yes, y_test_yes, **kw)
        mse_p.append(r2[-1][-1])

    def mean_list(l):
        return sum(l)/len(l)

    print(f'Media del error medio de todas las pruebas de un modelo lineal: {mean_list(mse_n)}')
    print(f'Media del error medio de todas las pruebas de un modelo polinomial: {mean_list(mse_p)}')
    
if __name__ == '__main__':
    run()