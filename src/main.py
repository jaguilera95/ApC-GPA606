import pandas as pd
import numpy as np
import seaborn as snb

DATA_DIR = 'data/insurance.csv'

def read_database(dir : str):
    return pd.read_csv(dir, delimiter= ',')

def run():
    df = read_database(DATA_DIR)
    print(df.head(5))

if __name__ == '__main__':
    run()
#hello

pei
