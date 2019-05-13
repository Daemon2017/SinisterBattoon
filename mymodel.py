from keras import backend as K
from keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, add, \
    Activation, concatenate
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam

K.set_image_dim_ordering('tf')

matrix_width = 25
matrix_height = 25
classes = 54

epochs_num = 10


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def conv3_block(input, size):
    conv = Conv2D(size, (1, 1), padding='same')(input)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)

    conv = Conv2D(size, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    return conv


def conv9_block(input, size):
    conv = Conv2D(size, (1, 1), padding='same')(input)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size, (9, 9), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = Conv2D(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = BatchNormalization(axis=3)(conv)
    return conv


def deconv3_block(input, size):
    conv = Conv2DTranspose(size, (1, 1), padding='same')(input)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size, (3, 3), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)

    conv = Conv2DTranspose(size, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size, (3, 3), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    return conv


def deconv9_block(input, size):
    conv = Conv2DTranspose(size, (1, 1), padding='same')(input)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size, (9, 9), padding='same')(conv)
    conv = Activation("elu")(conv)
    conv = Conv2DTranspose(size * 4, (1, 1), padding='same')(conv)
    conv = Activation("elu")(conv)
    return conv


def build():
    print('Building model...')
    filters = 8
    inputs = Input(shape=(matrix_height, matrix_width, 1))

    block1_9in = conv9_block(inputs, filters)
    block1_3in = conv3_block(inputs, filters)
    block2_3in = conv3_block(block1_3in, filters)
    block1_addin = concatenate([block2_3in, block1_9in])

    block2_9in = conv9_block(block1_addin, filters)
    block3_3in = conv3_block(block1_addin, filters)
    block4_3in = conv3_block(block3_3in, filters)
    block2_addin = concatenate([block4_3in, block2_9in])

    block3_9in = conv9_block(block2_addin, filters)
    block5_3in = conv3_block(block2_addin, filters)
    block6_3in = conv3_block(block5_3in, filters)
    block3_addin = concatenate([block6_3in, block3_9in])

    block3_9out = deconv9_block(block3_addin, filters)
    block6_3out = deconv3_block(block3_addin, filters)
    block5_3out = deconv3_block(concatenate([block5_3in, block6_3out]), filters)
    block3_addout = concatenate([block5_3out, block3_9out])

    block2_9out = deconv9_block(block3_addout, filters)
    block4_3out = deconv3_block(concatenate([block4_3in, block3_addout]), filters)
    block3_3out = deconv3_block(concatenate([block3_3in, block4_3out]), filters)
    block2_addout = concatenate([block3_3out, block2_9out])

    block1_9out = deconv9_block(block2_addout, filters)
    block2_3out = deconv3_block(concatenate([block2_3in, block2_addout]), filters)
    block1_3out = deconv3_block(concatenate([block1_3in, block2_3out]), filters)
    block1_addout = concatenate([block1_3out, block1_9out])

    output = Conv2DTranspose(classes, (1, 1), activation='softmax')(block1_addout)
    model = Model(inputs=[inputs],
                  outputs=[output])
    model.compile(optimizer=Adam(),
                  loss=categorical_crossentropy,
                  metrics=[dice_coef])
    print('Model is ready!')
    print(model.summary())
    return model
