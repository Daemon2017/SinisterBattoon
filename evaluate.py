import os
import numpy as np

from numpy import genfromtxt
from mymodel import build, matrix_width, matrix_height


def func_predict():
    print('Preparing prediction set...')
    files = os.listdir('./predict_input/')
    x_files_names = list(filter(lambda x: x.endswith('.csv'), files))
    total = len(x_files_names)

    x_predict = np.ndarray((total, matrix_height, matrix_width, 1), dtype=np.float)
    i = 0
    for x_file_name in x_files_names:
        img = genfromtxt('./predict_input/' + x_file_name, delimiter=',')
        x_predict[i] = np.array([img]).reshape(matrix_height, matrix_width, 1)
        x_predict[i][np.isnan(x_predict[i])] = 0
        for n in range(0, len(x_predict[i])):
            x_predict[i][n][n] = 6800
        i += 1
    print('Prediction set prepared!')

    x_predict = x_predict.astype('float32')
    x_predict /= 6800.0

    predictions = model.predict_on_batch(x_predict)
    i = 0
    for prediction in predictions:
        prediction = np.argmax(prediction, 2)
        short_name = os.path.splitext(x_files_names[i])[0]
        np.savetxt('./predict_output/' + str(short_name) + '.csv', prediction, fmt="%s", delimiter=',')
        i += 1


if not os.path.exists('predict_input'):
    os.makedirs('predict_input')
if not os.path.exists('predict_output'):
    os.makedirs('predict_output')

model = build()
model.load_weights('weights_batch.h5')
func_predict()
