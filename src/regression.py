'''
    Clase auxiliar para el manejo de los datos
    
    Autores: Juan Aguilera Toro
             Juan Manuel Camara Diaz
             Raul Salinas Natal
             
    Fecha: 2021-10-27
'''

import numpy as np


class Regression():

    def __init__(self, X: np.array, y: np.array, lr=0.01, regulador=0.001) -> None:
        """
        Inicializa el modelo de regresión lineal,
        con los datos de entrenamiento,
        y los parámetros de aprendizaje.

        Args:
            X (np.array): Matriz de datos de entrenamiento.
            y (np.array): Columna objetivo de entrenamiento.
            lr (float, optional): Tasa de aprendizaje. Defaults to 0.01.
            regulador (float, optional): Regularizador. Defaults to 0.001.
        """
        self.X = X
        if X[:, 0].max() == 1 and X[:, 0].min():
            X[:, 0] = 0
        self.y = y
        self.lr = lr
        self.regulador = regulador
        self.b = 0
        self.w = np.zeros(X.shape[1])
        self.mse = []

    def predict(self, X: np.array) -> np.array:
        """
        Predice el valor de la columna objetivo

        Args:
            X (np.array): Matriz de datos de prueba.

        Returns:
            np.array: Valor de la columna objetivo.
        """
        return (X @ self.w) + self.b

    def __hipotesis(self) -> np.array:
        """
        Calcula la hipótesis de la regresión lineal.
        """
        return (self.X @ self.w) + self.b

    def mse_lambda(self) -> float:
        """
        Calcula el error cuadrático medio. Con un regularizador.

        Returns:
            float: Error cuadrático medio.
        """
        return (1/(2 * self.X.shape[0])) * np.sum((self.__hipotesis() - self.y)**2) + (self.regulador * np.sum(self.w**2))

    def train(self, max_iter=10000, epsilon=0.1) -> None:
        """
        Entrena el modelo de regresión lineal, utilizando el algoritmo de gradiente descendente.
        Utiliza un criterio de parada para el aprendizaje.
        Tiene implementeado un regularizador.

        Args:
            max_iter (int, optional): Numero maximo de iteraciones. Defaults to 10000.
            epsilon (float, optional): Umbral de parada. Defaults to 0.1.
        """
        self.mse = [np.Inf]
        m = self.X.shape[0]

        for _ in range(max_iter):
            prediccion = self.__hipotesis()
            for j in range(self.w.shape[0]):
                regularizador = (self.regulador/m) * self.w[j]
                error = np.sum(((prediccion - self.y) * self.X[:, j]))/m
                self.w[j] = self.w[j] - self.lr * (error - regularizador)
            self.b = self.b - (self.lr * (2/m)) * np.sum(prediccion - self.y)
            self.mse.append(self.mse_lambda())
            if abs(self.mse[-1] - self.mse[-2]) < epsilon:
                break
