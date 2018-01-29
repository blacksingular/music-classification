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

# normalize simply
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

#append one
X_scaled_train = np.ones((456, 14))
X_scaled_train[:, 1:14] = X_scaled_1
X_scaled_test = np.ones((50, 14))
X_scaled_test[:, 1:14] = X_scaled_2
Y_scaled_train = y_train.reshape(456, 1)
Y_scaled_test = y_test.reshape(50, 1)


#Initialize
step_SGD = 5e-4
step_BGD = 5e-2
w_old_BGD = 0.2 * np.random.random_sample((14, 1))-0.1
w_old_SGD = 0.2 * np.random.random_sample((1, 14))-0.1
training_error_MSE_BGD = np.zeros(500)
training_error_MSE_SGD = np.zeros(500)
training_error_SGD = np.zeros(456)

#BGD 1(c)
for i in range(500):
    training_error_BGD = X_scaled_train.dot(w_old_BGD) - Y_scaled_train
    gradient_BGD = X_scaled_train.T.dot(training_error_BGD)
    gradient_BGD = 1.0/456.0 * gradient_BGD
    w_new_BGD = w_old_BGD - step_BGD * gradient_BGD
    w_old_BGD = w_new_BGD
    training_error_MSE_BGD[i] = (1.0/456.0)*sum((X_scaled_train.dot(w_new_BGD) - Y_scaled_train)**2)
#SGD 1(b)
for i in range(500):
    for j in range(456):
        training_error_SGD[j] = np.dot(w_old_SGD, X_scaled_train[j]) - Y_scaled_train[j]
        gradient_SGD = training_error_SGD[j] * X_scaled_train[j]
        w_new_SGD = np.subtract(w_old_SGD, step_SGD * gradient_SGD)
        w_old_SGD = w_new_SGD
    training_error_MSE_SGD[i] = (1.0/456.0)*sum((X_scaled_train.dot(w_new_SGD.T) - Y_scaled_train)**2)
#errors and bias
bias_SGD = w_new_SGD[0][0]
bias_BGD = w_new_BGD[0]
final_training_error_MSE_BGD = training_error_MSE_BGD[499]
final_training_error_MSE_SGD = training_error_MSE_SGD[499]
final_test_error_MSE_SGD = 1.0/50.0*sum((X_scaled_test.dot(w_new_SGD.T) - Y_scaled_test)**2)
final_test_error_MSE_BGD = 1.0/50.0*sum((X_scaled_test.dot(w_new_BGD) - Y_scaled_test)**2)

#closed form 1(d)
U, s, V = np.linalg.svd(X_scaled_train)
S_inv = np.zeros((14, 456))
S_inv[:, :14] = np.diag(1/s)
x_pseudoinverse = np.dot(V.T, np.dot(S_inv, U.T))
w_pred = x_pseudoinverse.dot(Y_scaled_train)
bias = w_pred[0]
training_error_closedform = 1.0/456.0*sum((X_scaled_train.dot(w_pred) - Y_scaled_train)**2)
test_error_closedform = 1.0/50.0*sum((X_scaled_test.dot(w_pred) - Y_scaled_test)**2)
#1(e)
# Original features
features_orig = dataset.data
labels_orig = dataset.target
Ndata = len(features_orig)
train_errs = []
test_errs = []
for k in range(100):
# Shuffle data
    rand_perm = np.random.permutation(Ndata)
    features = [features_orig[ind] for ind in rand_perm]
    labels = [labels_orig[ind] for ind in rand_perm]
# Train/test split
    Nsplit = 50
    X_train, y_train = features[:-Nsplit], labels[:-Nsplit]
    X_test, y_test = features[-Nsplit:], labels[-Nsplit:]
    # Preprocess your data - Normalization, adding a constant feature
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = np.hstack((np.ones((456, 1)), X_train))
    X_test = scaler.transform(X_test)
    X_test = np.hstack((np.ones((50, 1)), X_test))
    # Solve for optimal w
    # Use your solver function
    X_inv = np.linalg.pinv(X_train)
    w = X_inv.dot(y_train)

    # Collect train and test errors
    # Use your implementation of the mse function
    train_errs.append(sum((X_train.dot(w)-y_train)**2)/456.0)
    test_errs.append(sum((X_test.dot(w)-y_test)**2)/50.0)
print('Mean training error: ', np.mean(train_errs))
print('Mean test error: ', np.mean(test_errs))
