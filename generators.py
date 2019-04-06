import numpy as np

from keras.utils.np_utils import to_categorical
from numpy import genfromtxt
from mymodel import matrix_width, matrix_height, classes

size_of_batch = 8

start = 0
end = size_of_batch

start_test = 0
end_test = size_of_batch


def batch_train_generator(total, x_train_files_names, y_train_files_names):
    global start, end

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


def batch_test_generator(total_test, x_test_files_names, y_test_files_names):
    global start_test, end_test

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
