import os

from mymodel import build
from generators import batch_test_generator, size_of_batch, start_test, end_test


def func_test():
    print('Testing...')
    scores = model.evaluate_generator(
        generator=batch_test_generator(start_test, end_test, total_test, x_test_files_names, y_test_files_names),
        steps=total_test / size_of_batch,
        verbose=1)
    print('Loss: %2.2f, DICE: %2.2f' % (scores[0], scores[1]))
    print('Testing ended!')


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
model.load_weights('weights_batch.h5')
func_test()
