from keras import backend as K
from keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, add, Activation
from keras.models import Model
from keras.optimizers import Adam
from segmentation_models.losses import cce_dice_loss
from segmentation_models.metrics import f1_score

K.set_image_dim_ordering('tf')

matrix_width = 25
matrix_height = 25
classes = 90

epochs_num = 10


def conv_block(input, size):
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


def deconv_block(input, size):
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


def build():
    print('Building model...')
    filters = 8
    inputs = Input(shape=(matrix_height, matrix_width, 1))

    block1_in = conv_block(inputs, filters)
    block2_in = conv_block(block1_in, filters)
    block3_in = conv_block(block2_in, filters)
    block4_in = conv_block(block3_in, filters)
    block5_in = conv_block(block4_in, filters)

    block6_in = conv_block(block5_in, filters)
    block6_out = deconv_block(block6_in, filters)

    block5_out = deconv_block(add([block5_in, block6_out]), filters)
    block4_out = deconv_block(add([block4_in, block5_out]), filters)
    block3_out = deconv_block(add([block3_in, block4_out]), filters)
    block2_out = deconv_block(add([block2_in, block3_out]), filters)
    block1_out = deconv_block(add([block1_in, block2_out]), filters)

    output = Conv2DTranspose(classes, (1, 1), activation='softmax')(block1_out)
    model = Model(inputs=[inputs],
                  outputs=[output])
    model.compile(optimizer=Adam(),
                  loss=cce_dice_loss,
                  metrics=[f1_score])
    print('Model is ready!')
    print(model.summary())
    return model
