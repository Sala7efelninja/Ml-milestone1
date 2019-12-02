import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pre_processing as pp

# Loading data
data = pd.read_csv('Predicting_Mobile_App_Success.csv')
#data.drop([6941, 12624, 18477], inplace=True)
data = data[~data['Size'].isin(['Varies with device'])]
data.dropna(axis=0,how='any',inplace=True)
data=pp.remove_symbols(data)

columns_to_be_validated=['Rating','Price','Size','Reviews','Installs']
data=pp.delete_noise_data(data,columns_to_be_validated)
data=pp.to_float(data,columns_to_be_validated)

#X = data .loc[:, data.columns != 'Rating']

X = data.iloc[:,[1,3,4,5,6,7,9]]

Y = data['Rating']

# pre-processing

columns_to_be_transfomered = ['Category', 'Minimum Version', 'Content Rating']
columns_to_be_scaled = ['Installs', 'Reviews']
#X=pp.label_encoder_trans(X,columns_to_be_transfomered)


#dummy=pp.one_hot_trans(data,['Latest Version'])


encodedFeatures =pp.one_hot_trans(X,columns_to_be_transfomered)
scaled_columns=pp.feature_scaling(X,columns_to_be_scaled)

X.drop(columns=columns_to_be_transfomered, inplace=True)
X.drop(columns=columns_to_be_scaled, inplace=True)
X = np.array(X)
features = np.concatenate((X, encodedFeatures,scaled_columns), axis=1)


features = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.30, shuffle=True)
#Get the correlation between the features


multiLinearRegModel = linear_model.LinearRegression()
multiLinearRegModel.fit(X_train, y_train)
prediction = multiLinearRegModel.predict(X_test)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))
true_player_value = np.asarray(y_test)[0]
predicted_player_value = prediction[0]
print('True rate for the first application  in the test set  is : ' + str(true_player_value))
print('Predicted rate for the  first application  in the test set  is : ' + str(predicted_player_value))


