import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Load dataset
dataset = datasets.load_boston()

features = dataset.data
labels = dataset.target

Nsplit = 50
# Training set
X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
# Test set
X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
# col_mean = np.mean(X_train, axis=0)
# demeaned_matrix_train = X_train - col_mean
# demeaned_matrix_test = X_test - col_mean
# col_dev = np.std(demeaned_matrix_train, axis=0)
# X_scaled_1 = demeaned_matrix_train/col_dev
# X_scaled_2 = demeaned_matrix_test/col_dev
#
# X_scaled_train = np.ones((456, 14))
# X_scaled_train[:, :13] = X_scaled_1
# X_scaled_test = np.ones((50, 14))
# X_scaled_test[:, :13] = X_scaled_2
# X_reg_train = X_scaled_train[:-int(X_scaled_train.shape[0]*0.1)]
# X_reg_val = X_scaled_train[-int(X_scaled_train.shape[0]*0.1):]
# y_reg_train = y_train[:-int(X_scaled_train.shape[0]*0.1)]
# y_reg_val = y_train[-int(X_scaled_train.shape[0]*0.1):]

X_reg_train = X_train[:-int(X_train.shape[0]*0.1)]
X_reg_val = X_train[-int(X_train.shape[0]*0.1):]
y_reg_train = y_train[:-int(X_train.shape[0]*0.1)]
y_reg_val = y_train[-int(X_train.shape[0]*0.1):]
scaler = preprocessing.StandardScaler().fit(X_reg_train)
X_reg_train = scaler.transform(X_reg_train)
X_reg_train = np.hstack((np.ones((X_reg_train.shape[0], 1)), X_reg_train))
X_reg_val = scaler.transform(X_reg_val)
X_reg_val = np.hstack((np.ones((X_reg_val.shape[0], 1)), X_reg_val))
X_test = scaler.transform(X_test)
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

lamda = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
scaled_lamda = np.array([0.0, 41.1, 82.2, 123.3, 164.4, 205.5])
w = np.zeros((6, 14))
val_loss = np.zeros(6)
val_error = np.zeros(6)
for i in range(6):
    reg = scaled_lamda[i] * np.eye(X_reg_train.shape[1])
    inv = np.linalg.pinv(X_reg_train.T.dot(X_reg_train) + reg)
    w[i] = inv.dot(X_reg_train.T).dot(y_reg_train)
    val_loss[i] = (sum((y_reg_val - X_reg_val.dot(w[i]))**2)/45.0 + scaled_lamda[i]*sum(w[i]**2))/2.0
    val_error[i] = np.sqrt(sum((y_reg_val - X_reg_val.dot(w[i]))**2)/45.0)
test_error = np.sqrt(sum((y_test - X_test.dot(w[3]))**2)/50.0)