import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from simple_ml.nn.model import Model, Sequential
from simple_ml.nn.layer import Dense, Softmax, Input, Dropout, Activation, MaxPooling2D, AvgPool2D, AvgPooling2D, \
    Flatten
from simple_ml.nn.layer import Conv2d
from simple_ml.nn.optimizer import SGD, Momentum, Adam, RMSProp
from simple_ml.nn.initializer import zeros, ones
from simple_ml.utils.metric import accuracy, mean_absolute_error


def read_data(data_path='..\\tmp\\exp3', size=32, val_split=0.1, test_split=0.2, seed=1234):
    face_img_root = os.path.join(data_path, 'face')
    non_img_root = os.path.join(data_path, 'nonface')
    faces_img_paths = os.listdir(face_img_root)
    non_img_paths = os.listdir(non_img_root)
    faces_img_paths = [os.path.join(face_img_root, i) for i in faces_img_paths]
    non_img_paths = [os.path.join(non_img_root, i) for i in non_img_paths]

    root_name = 'x'.join([str(size), str(size)])
    root_path = os.path.join(data_path, root_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    all_Xy_data_path = os.path.join(root_path, 'data.bin')

    if not os.path.exists(all_Xy_data_path):
        # get face images
        face_imgs = []
        for face_img in tqdm(faces_img_paths):
            im = Image.open(face_img, mode='r')
            im = im.resize((size, size), Image.ANTIALIAS)
            face_imgs.append(np.asarray(im))
        face_arrays = np.array(face_imgs)
        face_labels = np.ones((face_arrays.shape[0], 1))
        # get non face images
        non_imgs = []
        for non_img in tqdm(non_img_paths):
            im = Image.open(non_img, mode='r')
            im = im.resize((size, size), Image.ANTIALIAS)
            non_imgs.append(np.asarray(im))
        non_arrays = np.array(non_imgs)
        non_labels = np.zeros((non_arrays.shape[0], 1))

        X = np.concatenate([face_arrays, non_arrays], axis=0)
        y = np.concatenate([face_labels, non_labels], axis=0)

        rand_idx = [_ for _ in range(X.shape[0])]
        random.seed(seed)
        random.shuffle(rand_idx)
        X = X[rand_idx]
        y = y[rand_idx]
        total_size = X.shape[0]
        train_size = int(total_size * (1 - val_split - test_split))
        val_size = train_size + int(total_size * val_split)
        X_mean = np.mean(X[:train_size], axis=(0, 1, 2), keepdims=True)
        X_train, y_train = (X[:train_size] - X_mean) / 255.0, y[:train_size]
        X_val, y_val = (X[train_size:val_size] - X_mean) / 255.0, y[train_size:val_size]
        X_test, y_test = (X[val_size:] - X_mean) / 255.0, y[val_size:]
        pickle.dump([X_train, y_train, X_val, y_val, X_test, y_test],
                    file=open(all_Xy_data_path, mode='wb'))
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(file=open(all_Xy_data_path, mode='rb'))
    print('X train: ', X_train.shape, ' y train: ', y_train.shape)
    print('X val: ', X_val.shape, ' y val: ', y_val.shape)
    print('X test: ', X_test.shape, ' y test: ', y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


def seq_cnn_face():
    X_train, y_train, X_val, y_val, X_test, y_test = read_data(size=28)
    print(
        'train set positive class portion: %.2f (%d / %d)' % (np.mean(y_train), int(np.sum(y_train)), y_train.shape[0]))
    print('val set positive class portion: %.2f (%d / %d)' % (np.mean(y_val), int(np.sum(y_val)), y_val.shape[0]))
    print('test set positive class portion: %.2f (%d / %d)' % (np.mean(y_test), int(np.sum(y_test)), y_test.shape[0]))

    model = Sequential()
    model.add(Input(batch_input_shape=(None, *X_train.shape[1:])))
    model.add(Conv2d(3, 16, stride=1, padding=2, activation='swish'))
    # model.add(MaxPooling2D(4, stride=2))
    # model.add(AvgPooling2D(4, stride=2))
    model.add(Conv2d(2, 32, stride=1, padding=0, activation='swish'))
    # model.add(MaxPooling2D(3, stride=2))
    # model.add(AvgPooling2D(3, stride=2))
    model.add(Conv2d(1, 64, stride=1, padding=0, activation='swish'))

    model.add(Flatten())
    model.add(Softmax(2))
    model.compile('CE', optimizer=Adam(lr=1e-3))
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=256, verbose=1, epochs=100,
              shuffle=True,
              metric='Accuracy', peek_type='single-cls')
    plt.subplot(211)
    plt.plot(model.train_losses, label='train_losses')
    plt.plot(model.validation_losses, label='valid_losses')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.subplot(212)
    plt.plot(model.train_metrics, label='train_accuracy')
    plt.plot(model.validation_metrics, label='valid_accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    seq_cnn_face()
