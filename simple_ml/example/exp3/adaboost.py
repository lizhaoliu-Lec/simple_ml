import sys

sys.path.append('./')
import os
import random
import pickle
import numpy as np

from tqdm import tqdm
from PIL import Image

from sklearn.tree import DecisionTreeClassifier
from simple_ml.preprocessing import NPDFeature
from simple_ml.ensemble import AdaBoostClassifier


def accuracy(y_t, y_pred):
    y_pred = y_pred.reshape((-1, 1))
    return np.mean(y_t == y_pred)


# data_path='..\\tmp\\exp3'
def convert_data_to_npd_feats(data_path='simple_ml/example/tmp/exp3', size=24, val_split=0.1, test_split=0.2,
                              seed=1234):
    face_img_root = os.path.join(data_path, 'face')
    non_img_root = os.path.join(data_path, 'nonface')
    faces_img_paths = os.listdir(face_img_root)
    non_img_paths = os.listdir(non_img_root)
    faces_img_paths = [os.path.join(face_img_root, i) for i in faces_img_paths]
    non_img_paths = [os.path.join(non_img_root, i) for i in non_img_paths]

    root_name = 'x'.join([str(size), str(size)])
    root_name = 'npd_' + root_name
    root_path = os.path.join(data_path, root_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    all_Xy_data_path = os.path.join(root_path, 'data.bin')

    if not os.path.exists(all_Xy_data_path):
        # get face images
        face_imgs = []
        for face_img in tqdm(faces_img_paths):
            im = Image.open(face_img, mode='r').convert('L')
            im = im.resize((size, size), Image.ANTIALIAS)
            im = np.asarray(im)
            npd = NPDFeature(image=im)
            im = npd.extract()
            face_imgs.append(im)
        face_arrays = np.array(face_imgs)
        face_labels = np.ones((face_arrays.shape[0], 1))
        # get non face images
        non_imgs = []
        for non_img in tqdm(non_img_paths):
            im = Image.open(non_img, mode='r').convert('L')
            im = im.resize((size, size), Image.ANTIALIAS)
            im = np.asarray(im)
            npd = NPDFeature(image=im)
            im = npd.extract()
            non_imgs.append(im)
        non_arrays = np.array(non_imgs)
        non_labels = -1 * np.ones((non_arrays.shape[0], 1))

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
        X_train, y_train = X[:train_size] / 255.0, y[:train_size]
        X_val, y_val = X[train_size:val_size] / 255.0, y[train_size:val_size]
        X_test, y_test = X[val_size:] / 255.0, y[val_size:]
        pickle.dump([X_train, y_train, X_val, y_val, X_test, y_test],
                    file=open(all_Xy_data_path, mode='wb'))
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(file=open(all_Xy_data_path, mode='rb'))
    print('X train: ', X_train.shape, ' y train: ', y_train.shape)
    print('X val: ', X_val.shape, ' y val: ', y_val.shape)
    print('X test: ', X_test.shape, ' y test: ', y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = convert_data_to_npd_feats()
    weak = DecisionTreeClassifier(max_depth=1)
    ad = AdaBoostClassifier(weak, 1)
    ad.fit(X_train, y_train, X_val, y_val, early_stop=True)
    # ad = weak
    ad.fit(X_train, y_train)
    y_pred = ad.predict(X_train)
    y_pred_val = ad.predict(X_val)
    y_pred_test = ad.predict(X_test)

    print('Train acc: ', accuracy(y_train, y_pred))
    print('Val acc: ', accuracy(y_val, y_pred_val))
    print('Test acc: ', accuracy(y_test, y_pred_test))
