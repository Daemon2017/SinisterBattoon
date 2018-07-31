import os

from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from generators import batch_test_generator, batch_train_generator, size_of_batch
from mymodel import build, epochs_num

K.set_image_dim_ordering('tf')

tbCallBack = TensorBoard(log_dir='./logs',
                         histogram_freq=0,
                         write_graph=True,
                         write_grads=True,
                         write_images=True)

checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_loss:.2f}.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='min',
                             period=1)


def func_train():
    print('Training...')
    model.fit_generator(generator=batch_train_generator(total, x_train_files_names, y_train_files_names),
                        validation_data=batch_test_generator(total_test, x_test_files_names, y_test_files_names),
                        epochs=epochs_num,
                        steps_per_epoch=total / size_of_batch,
                        validation_steps=total_test / size_of_batch,
                        verbose=1,
                        initial_epoch=0,
                        callbacks=[checkpoint,
                                   tbCallBack])
    print('Training ended!')


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
func_train()
