import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import pre_processing as pp

# Loading data
data = pd.read_csv('D:\sana 4\ML\Project\Predicting_Mobile_App_Success.csv')
data.drop([6941, 12624, 18477], inplace=True)
data = data[~data['Size'].isin(['Varies with device'])]
#X = data.loc[:, data.columns != 'Rating']
X = data.iloc[:,[1,3,4,5,6,7,9]]
Y = data['Rating']

# pre-processing
X=pp.remove_symbols(X)

columns_to_be_transfomered = ['Category', 'Minimum Version', 'Content Rating']
columns_to_be_scaled = ['Installs', 'Reviews']
X=pp.label_encoder_trans(X,columns_to_be_transfomered)

scaled_columns=pp.feature_scaling(X,columns_to_be_scaled)

X.drop(columns=columns_to_be_scaled, inplace=True)
X = np.array(X)
features = np.concatenate((X,scaled_columns), axis=1)


features = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.30, shuffle=True)

print('Poly model')
poly_features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
true_player_value = np.asarray(y_test)[0]
predicted_player_value = prediction[0]
print('True rate for the first application  in the test set  is : ' + str(true_player_value))
print('Predicted rate for the  first application  in the test set  is : ' + str(predicted_player_value))

