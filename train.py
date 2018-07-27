import numpy as np
from numpy import genfromtxt
import os
import tensorflow as tf
import scipy.misc
from skimage.io import imread
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose, Dropout, BatchNormalization, add, \
    AveragePooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback, LearningRateScheduler
from keras import backend as K
from keras.utils.np_utils import to_categorical
from mymodel import dice_coef, dice_coef_loss, build, matrix_width, matrix_height, classes, L_0, epochs_num

size_of_batch = 8

start = 0
end = size_of_batch

start_test = 0
end_test = size_of_batch

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_loss:.2f}.h5",
                             monitor=dice_coef,
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='max',
                             period=1)


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if (self.batch % self.N == 0) and (self.batch != 0):
            self.model.save_weights('weights_batch' + str(batch) + '.h5')
        self.batch += 1


class TB(TensorBoard):
    def __init__(self, log_every=1, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0

    def on_batch_end(self, batch, logs=None):
        self.counter += 1
        if self.counter % self.log_every == 0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        super().on_batch_end(batch, logs)


def batch_train_generator():
    global start, end, total, x_train_files_names, y_train_files_names

    batch_num = 0
    while True:
        print('\n------------------------------')
        print('Generating training batch ' + str(batch_num))
        x_train = np.ndarray((size_of_batch, matrix_height, matrix_width, 1), dtype=np.float)
        y_train = np.ndarray((size_of_batch, matrix_height, matrix_width, 1), dtype=np.float)

        sample = 0
        for j in range(start, end):
            print(
                'Preparing training file: #' + str(sample) + ', input name: ' + str(
                    x_train_files_names[j]) + ', output name: ' + str(
                    y_train_files_names[j]))
            x_matrix = genfromtxt('./train_input/' + x_train_files_names[j], delimiter=',')
            y_matrix = genfromtxt('./train_output/' + y_train_files_names[j], delimiter=',')
            x_train[sample] = np.array([x_matrix]).reshape(matrix_height, matrix_width, 1)
            y_train[sample] = np.array([y_matrix]).reshape(matrix_height, matrix_width, 1)
            sample += 1

        x_train = x_train.astype('float32')
        y_train = y_train.astype('float32')
        x_train /= 6800.0
        y_train = to_categorical(y_train, classes)

        print('Start is ' + str(start) + ', end is ' + str(end))
        start += size_of_batch
        end += size_of_batch
        if end > total:
            start = 0
            end = size_of_batch

        print('Training batch ' + str(batch_num) + ' generated!')
        batch_num += 1
        if batch_num == size_of_batch:
            batch_num = 0

        yield x_train, y_train


def batch_test_generator():
    global start_test, end_test, total_test, x_test_files_names, y_test_files_names

    batch_num = 0
    while True:
        print('\n------------------------------')
        print('Generating test batch ' + str(batch_num))
        x_test = np.ndarray((size_of_batch, matrix_height, matrix_width, 1), dtype=np.float)
        y_test = np.ndarray((size_of_batch, matrix_height, matrix_width, 1), dtype=np.float)

        sample = 0
        for j in range(start_test, end_test):
            print('Preparing test file: #' + str(sample) + ', input name: ' + str(
                x_test_files_names[j]) + ', output name: ' + str(
                y_test_files_names[j]))
            x_matrix = genfromtxt('./test_input/' + x_test_files_names[j], delimiter=',')
            y_matrix = genfromtxt('./test_output/' + y_test_files_names[j], delimiter=',')
            x_test[sample] = np.array([x_matrix]).reshape(matrix_height, matrix_width, 1)
            y_test[sample] = np.array([y_matrix]).reshape(matrix_height, matrix_width, 1)
            sample += 1

        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        x_test /= 6800.0
        y_test = to_categorical(y_test, classes)

        print('Start is ' + str(start_test) + ', end is ' + str(end_test))
        start_test += size_of_batch
        end_test += size_of_batch
        if end_test > total_test:
            start_test = 0
            end_test = size_of_batch

        print('Test batch ' + str(batch_num) + ' generated!')
        batch_num += 1
        if batch_num == size_of_batch:
            batch_num = 0

        yield x_test, y_test


def train():
    print('Training...')
    model.fit_generator(generator=batch_train_generator(),
                        validation_data=batch_test_generator(),
                        epochs=epochs_num,
                        steps_per_epoch=total / size_of_batch,
                        validation_steps=total_test / size_of_batch,
                        verbose=1,
                        initial_epoch=0,
                        callbacks=[checkpoint,
                                   # tbCallBack,
                                   # WeightsSaver(model, 1000),
                                   TB(1)])
    print('Training ended!')
    model.save('weights_batch.h5')
    Y_pred = model.predict_generator(generator=batch_test_generator(),
                                     steps=total_test / size_of_batch,
                                     verbose=1)
    y_pred = np.argmax(Y_pred,
                       axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(batch_test_generator().classes,
                           y_pred))
    print('Classification Report')
    print(classification_report(batch_test_generator().classes,
                                y_pred))


if not os.path.exists('logs'):
    os.makedirs('logs')

if not os.path.exists('train_input'):
    os.makedirs('train_input')
if not os.path.exists('train_output'):
    os.makedirs('train_output')
x_files = os.listdir('./train_input/')
y_files = os.listdir('./train_output/')
x_train_files_names = list(filter(lambda x: x.endswith('.csv'), x_files))
y_train_files_names = list(filter(lambda x: x.endswith('.csv'), y_files))
x_train_files_names = sorted(x_train_files_names)
y_train_files_names = sorted(y_train_files_names)
x_total = len(x_train_files_names)
y_total = len(y_train_files_names)
total = 0
if x_total == 0:
    print('Quantity of X or Y in train set is equals to zero. Work stopped!')
    exit()
else:
    if x_total != y_total:
        print('Quantity of X and Y in train set differs. Work stopped!')
        exit()
    else:
        total = x_total
        print('Quantity of X and Y in train set is the same. Work continues!')

if not os.path.exists('test_input'):
    os.makedirs('test_input')
if not os.path.exists('test_output'):
    os.makedirs('test_output')
x_test_files = os.listdir('./test_input/')
y_test_files = os.listdir('./test_output/')
x_test_files_names = list(filter(lambda x: x.endswith('.csv'), x_test_files))
y_test_files_names = list(filter(lambda x: x.endswith('.csv'), y_test_files))
x_test_files_names = sorted(x_test_files_names)
y_test_files_names = sorted(y_test_files_names)
x_test_total = len(x_test_files_names)
y_test_total = len(y_test_files_names)
total_test = 0
if x_test_total == 0:
    print('Quantity of X or Y in test set is equals to zero. Work stopped!')
    exit()
else:
    if x_test_total != y_test_total:
        print('Quantity of X and Y in test set differs. Work stopped!')
        exit()
    else:
        total_test = x_test_total
        print('Quantity of X and Y in test set is the same. Work continues!')

model = build()
train()
