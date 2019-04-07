from keras import backend as K
from keras.layers import Conv2D, Input, Conv2DTranspose, BatchNormalization, add, \
    SpatialDropout2D, Activation
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam

K.set_image_dim_ordering('tf')

L_0 = 0.0001

matrix_width = 25
matrix_height = 25
classes = 56

epochs_num = 100


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


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
    conv = SpatialDropout2D(0.2)(input)
    conv = Conv2DTranspose(size, (1, 1), padding='same')(conv)
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
    filters = 32
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
    model.compile(optimizer=Adam(lr=L_0),
                  loss=dice_coef_loss,
                  metrics=[dice_coef, f1])
    print('Model is ready!')
    print(model.summary())
    return model
