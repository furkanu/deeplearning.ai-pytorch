import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_happy.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]).transpose((0, 3, 1, 2)) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_happy.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]).transpose((0, 3, 1, 2)) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    X_train_numpy = train_set_x_orig / 255.
    X_test_numpy = test_set_x_orig / 255.
    
    Y_train_numpy = train_set_y_orig.reshape((train_set_y_orig.shape[0], 1))
    Y_test_numpy = test_set_y_orig.reshape((test_set_y_orig.shape[0], 1))
    
    return X_train_numpy, Y_train_numpy, X_test_numpy, Y_test_numpy, classes

