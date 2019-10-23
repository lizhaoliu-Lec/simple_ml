import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from simple_ml.nn.model import Model, Sequential
from simple_ml.nn.layer import Dense, Softmax, Input, Dropout, Activation, MaxPooling2D, AvgPool2D, AvgPooling2D, \
    Flatten
from simple_ml.nn.layer import Conv2d
from simple_ml.nn.optimizer import SGD, Momentum, Adam
from simple_ml.nn.initializer import zeros
from simple_ml.utils.metric import accuracy, mean_absolute_error


def read_data(data_path='tmp\\exp3'):
    face_img_root = os.path.join(data_path, 'face')
    non_img_root = os.path.join(data_path, 'nonface')
    faces_img_paths = os.listdir(face_img_root)
    non_img_paths = os.listdir(non_img_root)
    faces_img_paths = [os.path.join(face_img_root, i) for i in faces_img_paths]
    non_img_paths = [os.path.join(non_img_root, i) for i in non_img_paths]

    im = Image.open(faces_img_paths[-1], mode='r')
    im = im.resize((32, 32), Image.ANTIALIAS)
    im = np.asarray(im)
    print(type(im))
    print(im.shape)
    print(im)


def seq_cnn_face():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('tmp/data', one_hot=False)
    training_data = np.array([image.reshape(28, 28, 1) for image in mnist.train.images])
    training_label = mnist.train.labels
    valid_data = np.array([image.reshape(28, 28, 1) for image in mnist.validation.images])
    valid_label = mnist.validation.labels
    label_size = 10

    model = Sequential()
    model.add(Input(batch_input_shape=(None, 28, 28, 1)))
    model.add(Conv2d(3, 16, stride=1, padding=2, activation='relu'))
    # model.add(MaxPooling2D(4, stride=2))
    model.add(AvgPooling2D(4, stride=2))
    model.add(Conv2d(2, 32, stride=1, padding=0, activation='relu'))
    # model.add(MaxPooling2D(3, stride=2))
    model.add(AvgPooling2D(3, stride=2))
    model.add(Conv2d(1, 64, stride=1, padding=0, activation='relu'))
    # model.add(MaxPooling2D(3, stride=3))
    # model.add(AvgPooling2D(3, stride=3))

    model.add(Flatten())
    model.add(Softmax(label_size))
    model.compile('CE', optimizer=Adam(lr=1e-3))
    model.fit(training_data, training_label, validation_data=(valid_data, valid_label),
              batch_size=256, verbose=1, epochs=2, metric='Accuracy', peek_type='single-cls')
    # model.fit(training_data[:1000], training_label[:1000], validation_data=(valid_data[:1000], valid_label[:1000]),
    #           batch_size=256, verbose=1, epochs=10, metric='Accuracy', peek_type='single-cls')
    # model.fit(training_data[:100], training_label[:100], validation_data=(valid_data[:50], valid_label[:50]),
    #           batch_size=256, verbose=10, epochs=100, metric='Accuracy', peek_type='single-cls')
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
    read_data()
