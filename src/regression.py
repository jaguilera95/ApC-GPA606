import numpy as np

class Regression():

    def __init__(self, X, y, lr = 0.01, regulador = 0.001) -> None:
        self.X = X
        if X[:,0].max() == 1 and X[:,0].min():
            X[:,0] = 0
        self.y = y
        self.lr = lr
        self.regulador = regulador
        self.b = 0
        self.w = np.zeros(X.shape[1])
        self.mse = []
    
    def predict(self, X):
        return (X @ self.w) + self.b

    def hipotesis(self):
        return (self.X @ self.w) + self.b

    def mse_lambda(self):
        return (1/(2 * self.X.shape[0])) * np.sum((self.hipotesis() - self.y)**2) + (self.regulador * np.sum(self.w**2))

    def train(self, max_iter = 10000, epsilon = 0.1):
        self.mse = [np.Inf]
        m = self.X.shape[0]
        
        for _ in range(max_iter):
            prediccion = self.hipotesis()
            for j in range(self.w.shape[0]):
                final = (self.regulador/m) * self.w[j]
                otro = np.sum(((prediccion - self.y) * self.X[:,j]))/m
                self.w[j] = self.w[j] - self.lr * (otro - final)
            self.b = self.b - (self.lr * (2/m)) * np.sum(prediccion - self.y)
            self.mse.append(self.mse_lambda())
            if abs(self.mse[-1] - self.mse[-2]) < epsilon:
                break