import h5py
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, TensorDataset


def get_data(batch_size=64):
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    x_train = np.array(train_dataset["train_set_x"][:]) # your train set features
    x_train = np.transpose(x_train, (0, 3, 1, 2))
    y_train = np.array(train_dataset["train_set_y"][:]) # your train set labels
    y_train = y_train.reshape((1, y_train.shape[0])).T

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    x_test = np.array(test_dataset["test_set_x"][:]) # your test set features
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    y_test = np.array(test_dataset["test_set_y"][:]) # your test set labels
    y_test = y_test.reshape((1, y_test.shape[0])).T

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    X_train_tensor = torch.tensor(x_train, dtype=torch.float)/255
    Y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(x_test, dtype=torch.float)/255
    Y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    return train_dataset, test_dataset, train_loader, test_loader, classes


def path_to_input(image_path, input_size, device):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_size, input_size))                  #Resize
    img = img[..., ::-1].transpose((2, 0, 1))                        #BGR -> RGB and HxWxC -> CxHxW
    img = img[np.newaxis, ...] / 255.0                               #Add a channel at 0, thus making it a batch
    img = torch.tensor(img, dtype=torch.float, device=device)        #Convert to Tensor
    return img
