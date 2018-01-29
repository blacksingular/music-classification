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

# normalize
col_mean = np.mean(X_train, axis=0)
demeaned_matrix_train = X_train - col_mean
demeaned_matrix_test = X_test - col_mean
col_dev = np.std(demeaned_matrix_train, axis=0)
X_scaled_1 = demeaned_matrix_train/col_dev
X_scaled_2 = demeaned_matrix_test/col_dev
X_scaled_train = np.ones((456, 14))
X_scaled_train[:, 1:14] = X_scaled_1
X_scaled_test = np.ones((50, 14))
X_scaled_test[:, 1:14] = X_scaled_2
Y_scaled_train = y_train.reshape(456, 1)
Y_scaled_test = y_test.reshape(50, 1)


test_error = []
train_error = []
X_new_train = np.ones((456, 1))
X_new_test = np.ones((50, 1))

#2(a)
# 0 order

X_inv = np.linalg.pinv(X_new_train)
w = X_inv.dot(y_train)
train_error.append(np.sqrt(sum((y_train-X_new_train.dot(w))**2)/456.0))
test_error.append(np.sqrt(sum((y_test-X_new_test.dot(w))**2)/50.0))
print w

#1~4 order
for i in range(1, 5):
    X_new_train = np.hstack((X_new_train, X_train**i))
    X_new_test = np.hstack((X_new_test, X_test**i))
    X_inv = np.linalg.pinv(X_new_train)
    w = X_inv.dot(y_train)
    train_error.append(np.sqrt(sum((y_train-X_new_train.dot(w))**2)/456.0))
    test_error.append(np.sqrt(sum((y_test-X_new_test.dot(w))**2)/50.0))
print w

#2(b) proportion

prop = np.array([0.2, 0.4, 0.6, 0.8, 1])
train_error_p = np.zeros(5)
test_error_p = np.zeros(5)
X_train_p = []
X_test_p = []
for i in range(5):
    # index = int(X_train.shape[0]*prop[i])
    # X_train_p = np.ones((index, 14))
    # X_train_p[:, :13] = X_train[:index, :]
    # X_test_p = np.ones((50, 14))
    # X_test_p[:, :13] = X_test
    # scaler = preprocessing.StandardScaler().fit(X_train_p)
    # X_train_p = scaler.transform(X_train_p)
    # X_test_p = scaler.transform(X_test_p)
    # y_train_p = y_train[:index]
    # X_inv = np.linalg.pinv(X_train_p)
    # w_pred = X_inv.dot(y_train_p)
    # y_pred_train = X_train_p.dot(w_pred)
    # y_pred_test = X_test_p.dot(w_pred)
    # train_error_p[i] = np.sqrt(sum((y_train_p-y_pred_train)**2)/X_train_p.shape[0])
    # test_error_p[i] = np.sqrt(sum((y_test-y_pred_test)**2)/50.0)
    index = int(X_train.shape[0]*prop[i])
    X_train_p = X_train[:index, :]
    scaler = preprocessing.StandardScaler().fit(X_train_p)
    X_train_p = scaler.transform(X_train_p)
    X_test_p = scaler.transform(X_test)
    X_train_p = np.hstack((np.ones((index, 1)), X_train_p))
    X_test_p = np.hstack((np.ones((50, 1)), X_test_p))
    y_train_p = y_train[:index]
    X_inv = np.linalg.pinv(X_train_p)
    w_pred = X_inv.dot(y_train_p)
    y_pred_train = X_train_p.dot(w_pred)
    y_pred_test = X_test_p.dot(w_pred)
    train_error_p[i] = np.sqrt(sum((y_train_p-y_pred_train)**2)/X_train_p.shape[0])
    test_error_p[i] = np.sqrt(sum((y_test-y_pred_test)**2)/50.0)