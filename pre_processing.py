
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
def isfloat(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def delete_noise_data(data,cols):

    for i in cols:
        z = 0
        print(".........",len(data))
        for x in data[i]:

            if isfloat(x):
                data[i].values[z]=float(x)
            else:
                print(x,":",z,np.where(data[i]==x)[0])
                print(data[i].values[z])
                data.drop([z],inplace=True)
                print(x, ":", z, np.where(data[i] == x)[0])
                print(data[i].values[z-1])
                z=z-1
            z = z + 1

        print("..........>",len(data))


    return data


def remove_symbols(X):
    X['Price'] = X['Price'].str.replace('$', '')
    X['Installs'] = X['Installs'].str.replace(',', '')
    X['Installs'] = X['Installs'].str.replace('+', '')
    X['Size'] = X['Size'].str.replace(',', '')
    X['Size'] = [(lambda x: float(x[0:-1]) if x[-1] == 'M' else float(x[0:-1]) / 1000)(x) for x in X['Size']]
    return  X


def label_encoder_trans(X, columns_to_be_transfomered):
    labelEncoder = LabelEncoder()
    for s in columns_to_be_transfomered:
        X[s] = labelEncoder.fit_transform(X[s])
    return X


def one_hot_trans(X,columns_to_be_transfomered):

    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(categories='auto'), columns_to_be_transfomered)])
    encodedFeatures = colT.fit_transform(X).toarray()
    return  encodedFeatures

def feature_scaling(X,columns_to_be_scaled):
    colT = ColumnTransformer([('std', MinMaxScaler(), columns_to_be_scaled)])
    scaled_feature= colT.fit_transform(X)
    return scaled_feature


