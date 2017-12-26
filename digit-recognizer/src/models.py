# -*- coding: utf-8 -*-

import os
import sys

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.models import Sequential
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping


class dataProcess(object):
    def __init__(self, **kwargs):
        self.train_val_split = kwargs.get('train_val_split', 0.1)

    def load(self, dir):
        print('>>> loading data')
        train_data = pd.read_csv(os.path.join(dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(dir, 'test.csv'))
        return train_data, test_data

    def preprocess(self, train_data, test_data):
        print('>>> preprocessing data')
        trainx = train_data[train_data.columns[1:]].values / 255.0
        trainy = train_data['label'].values
        trainy = to_categorical(trainy, num_classes=10)
        testx = test_data.values / 255.0
        del train_data
        del test_data
        idx = np.arange(trainx.shape[0])
        np.random.shuffle(idx)
        trainx, trainy = trainx[idx], trainy[idx]
        valid_size = int(trainx.shape[0] * self.train_val_split)
        valx, valy = trainx[:valid_size], trainy[:valid_size]
        trainx, trainy = trainx[valid_size:], trainy[valid_size:]
        print('shape of train data: %s' % (str(trainx.shape)))
        print('shape of val data: %s' % (str(valx.shape)))
        print('shape of test data: %s' % (str(testx.shape)))
        print('count of label in train set')
        print((trainy == 1).sum(axis=0))
        print('count of label in validation set')
        print((valy == 1).sum(axis=0))
        return trainx, trainy, valx, valy, testx


class mlpModel(object):
    def __init__(self, **kwargs):
        self.dropout = kwargs.get('dropout', 0.5)
        self.activation = kwargs.get('activation', 'relu')
        self.batch_normalization = kwargs.get('batch_normalization', False)
        self.optimizer = kwargs.get('optimizer', 'SGD')
        self.lr = kwargs.get('lr', 0.01)
        self.rho = kwargs.get('rho', 0.9)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.log_dir = kwargs.get('log_dir', '../tmp/model_mlp')

        model_mlp = Sequential()
        model_mlp.add(Dense(2048, activation=self.activation, input_dim=784))
        model_mlp.add(Dropout(self.dropout))
        model_mlp.add(Dense(512, activation=self.activation))
        if self.batch_normalization:
            model_mlp.add(BatchNormalization())
        model_mlp.add(Dense(128, activation=self.activation))
        model_mlp.add(Dropout(self.dropout))
        model_mlp.add(Dense(10, activation='softmax'))
        if self.optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.lr, epsilon=1e-6, rho=self.rho, decay=0.0)
        elif self.optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.lr, epsilon=1e-6)
        else:
            optimizer = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model_mlp.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_mlp = model_mlp

    def fit(self, trainx, trainy, valx, valy):
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        history = self.model_mlp.fit(trainx, trainy,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size,
                                    validation_data=(valx, valy),
                                    callbacks=[TensorBoard(log_dir=self.log_dir), learning_rate_reduction],
                                    verbose=2)
        return history

    def predict(self, testx):
        testy = self.model_mlp.predict(testx)
        return testy

    def save_result(self, testy, out_file):
        ntest = testy.shape[0]
        testy = np.argmax(testy, axis=1)
        result = pd.DataFrame(data=np.zeros((ntest, 2), dtype='int'), columns=['ImageId', 'Label'])
        result['ImageId'] = np.arange(ntest) + 1
        result['Label'] = testy
        result.to_csv(out_file, index=False)

    def save_history(self, history, out_file):
        train_loss = history.history['loss']
        train_acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        res = pd.DataFrame(data=np.zeros((len(train_loss), 4)), columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        res['train_loss'] = train_loss
        res['train_acc'] = train_acc
        res['val_loss'] = val_loss
        res['val_acc'] = val_acc
        res.to_csv(out_file, index=False)


class cnnModel(object):
    def __init__(self, **kwargs):
        self.dropout = kwargs.get('dropout', 0.5)
        self.activation = kwargs.get('activation', 'relu')
        self.pool_strides = kwargs.get('pool_strides', 2)
        self.dense_size = kwargs.get('dense_size', 256)
        self.data_augmentation = kwargs.get('data_augmentation', False)
        self.batch_normalization = kwargs.get('batch_normalization', False)
        self.optimizer = kwargs.get('optimizer', 'SGD')
        self.lr = kwargs.get('lr', 0.001)
        self.rho = kwargs.get('rho', 0.9)
        self.early_stopping = kwargs.get('early_stopping', False)
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 100)
        self.log_dir = kwargs.get('log_dir', '../tmp/model_cnn')

        model_cnn = Sequential()
        if self.batch_normalization:
            model_cnn.add(BatchNormalization(input_shape=(28, 28, 1)))
            model_cnn.add(Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=self.activation))
        else:
            model_cnn.add(Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=self.activation, input_shape=(28, 28, 1)))
        model_cnn.add(Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=self.activation))
        model_cnn.add(Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=self.activation))
        model_cnn.add(MaxPool2D(pool_size=[2, 2], strides=self.pool_strides))
        model_cnn.add(Dropout(self.dropout))
        model_cnn.add(Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=self.activation))
        model_cnn.add(Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=self.activation))
        model_cnn.add(Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=self.activation))
        model_cnn.add(MaxPool2D(pool_size=[2, 2], strides=self.pool_strides))
        model_cnn.add(Dropout(self.dropout))
        model_cnn.add(Flatten())
        model_cnn.add(Dense(self.dense_size, activation=self.activation))
        model_cnn.add(Dropout(self.dropout))
        model_cnn.add(Dense(10, activation='softmax'))
        if self.optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.lr, epsilon=1e-6, rho=self.rho, decay=0.0)
        elif self.optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.lr, epsilon=1e-6)
        else:
            optimizer = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model_cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_cnn = model_cnn

    def fit(self, trainx, trainy, valx, valy):
        trainx = trainx.reshape(-1, 28, 28, 1)
        valx = valx.reshape(-1, 28, 28, 1)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        if self.data_augmentation:
            imagen = ImageDataGenerator(rotation_range=10,
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        zoom_range=0.1)
            imagen.fit(trainx)
            history = self.model_cnn.fit_generator(imagen.flow(trainx, trainy, batch_size=self.batch_size),
                                            epochs=self.epochs,
                                            steps_per_epoch=trainx.shape[0] // self.batch_size,
                                            validation_data=(valx, valy),
                                            verbose=2,
                                            callbacks=[TensorBoard(log_dir=self.log_dir), learning_rate_reduction])
        else:
            history = self.model_cnn.fit(trainx, trainy,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    validation_data=(valx, valy),
                                    callbacks=[TensorBoard(log_dir=self.log_dir), learning_rate_reduction],
                                    verbose=2)
        return history

    def predict(self, testx):
        testx = testx.reshape(-1, 28, 28, 1)
        testy = self.model_cnn.predict(testx)
        return testy

    def save_result(self, testy, out_file):
        ntest = testy.shape[0]
        testy = np.argmax(testy, axis=1)
        result = pd.DataFrame(data=np.zeros((ntest, 2), dtype='int'), columns=['ImageId', 'Label'])
        result['ImageId'] = np.arange(ntest) + 1
        result['Label'] = testy
        result.to_csv(out_file, index=False)

    def save_history(self, history, out_file):
        train_loss = history.history['loss']
        train_acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        res = pd.DataFrame(data=np.zeros((len(train_loss), 4)), columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        res['train_loss'] = train_loss
        res['train_acc'] = train_acc
        res['val_loss'] = val_loss
        res['val_acc'] = val_acc
        res.to_csv(out_file, index=False)


class lstmModel(object):
    def __init__(self, **kwargs):
        self.dropout = kwargs.get('dropout', 0.5)
        self.activation = kwargs.get('activation', 'relu')
        self.lstm_outdim = kwargs.get('lstm_outdim', 512)
        self.optimizer = kwargs.get('optimizer', 'SGD')
        self.lr = kwargs.get('lr', 0.01)
        self.rho = kwargs.get('rho', 0.9)
        self.batch_normalization = kwargs.get('batch_normalization', False)
        self.log_dir = kwargs.get('log_dir', '../tmp/model_lstm')
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 100)

        model_lstm = Sequential()
        model_lstm.add(LSTM(self.lstm_outdim, input_shape=(28, 28)))
        model_lstm.add(Dropout(self.dropout))
        if self.batch_normalization:
            model_lstm.add(BatchNormalization())
        model_lstm.add(Dense(128, activation=self.activation))
        model_lstm.add(Dropout(self.dropout))
        model_lstm.add(Dense(10, activation='softmax'))
        if self.optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.lr, epsilon=1e-6, rho=self.rho, decay=0.0)
        elif self.optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.lr, epsilon=1e-6)
        else:
            optimizer = SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model_lstm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model_lstm = model_lstm

    def fit(self, trainx, trainy, valx, valy):
        trainx = trainx.reshape(-1, 28, 28)
        valx = valx.reshape(-1, 28, 28)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        history = self.model_lstm.fit(trainx, trainy,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    validation_data=(valx, valy),
                                    callbacks=[TensorBoard(log_dir=self.log_dir), learning_rate_reduction],
                                    verbose=2)
        return history

    def predict(self, testx):
        testx = testx.reshape(-1, 28, 28)
        testy = self.model_lstm.predict(testx)
        return testy

    def save_result(self, testy, out_file):
        ntest = testy.shape[0]
        testy = np.argmax(testy, axis=1)
        result = pd.DataFrame(data=np.zeros((ntest, 2), dtype='int'), columns=['ImageId', 'Label'])
        result['ImageId'] = np.arange(ntest) + 1
        result['Label'] = testy
        result.to_csv(out_file, index=False)

    def save_history(self, history, out_file):
        train_loss = history.history['loss']
        train_acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        res = pd.DataFrame(data=np.zeros((len(train_loss), 4)), columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        res['train_loss'] = train_loss
        res['train_acc'] = train_acc
        res['val_loss'] = val_loss
        res['val_acc'] = val_acc
        res.to_csv(out_file, index=False)


if __name__ == '__main__':
    dp = dataProcess(train_val_split=0.15)
    train_data, test_data = dp.load('../data')
    trainx, trainy, valx, valy, testx = dp.preprocess(train_data, test_data)
    clf = cnnModel(epochs=40, dense_size=300, optimizer='RMSprop', data_augmentation=True)
    history = clf.fit(trainx, trainy, valx, valy)
    clf.save_history(history, '../tmp/history_17122301.csv')
    testy = clf.predict(testx)
    clf.save_result(testy, '../result/result_17122301.csv')
