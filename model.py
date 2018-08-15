from config import TRAIN_IMAGE_SIZE, IMAGE_CHANNELS
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Concatenate, Add, add, MaxPool2D, AveragePooling2D
from keras.models import Model as Model, load_model
from keras import optimizers, regularizers, callbacks
import keras.backend as K
from config import BOARD_SIZE, LR, MOMENTUM, EPOCHS, REG, BATCH_SIZE, VALIDATION_SPLIT, model_dir, MODEL_SAVE_PERIOD
import os
from misc import logger
import numpy as np
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def loss(y_true, y_pred):
    loss_list = []
    for i in range(BOARD_SIZE * BOARD_SIZE):
        loss = K.categorical_crossentropy(y_true[:, i*3:(i+1)*3], y_pred[:, i*3:(i+1)*3])
        loss_list.append(loss)
    total_loss = add(loss_list)
    return total_loss


def acc_stone(y_true, y_pred):
    correct = 0
    total = 0
    for i in range(BOARD_SIZE * BOARD_SIZE):
        true_max_index = K.argmax(y_true[:, i*3:(i+1)*3], axis=1)
        pred_max_index = K.argmax(y_pred[:, i*3:(i+1)*3], axis=1)
        mask = K.cast(K.equal(true_max_index, pred_max_index), 'int32')
        correct += K.sum(mask)
        total += K.sum(K.ones_like(mask))
    return correct / total


def acc_board(y_true, y_pred):
    equal_times = K.sum(K.zeros_like(y_true, dtype='int32'), axis=1)
    for i in range(BOARD_SIZE * BOARD_SIZE):
        true_max_index = K.argmax(y_true[:, i*3:(i+1)*3], axis=1)
        pred_max_index = K.argmax(y_pred[:, i*3:(i+1)*3], axis=1)
        mask = K.equal(true_max_index, pred_max_index)
        mask = K.cast(mask, 'int32')
        equal_times += mask
    corrects = K.sum(K.cast(K.equal(equal_times, BOARD_SIZE * BOARD_SIZE), 'int32'))
    total = K.sum(K.ones_like(equal_times))
    return corrects / total


class PZSZModel:
    _REGULARIZERS = {
        'kernel_regularizer': regularizers.l2(REG),
        # 'bias_regularizer': regularizers.l2(conf['c']),
    }
    _RES_BLOCK_COUNTS = [2, 2, 3, 2]
    _RES_BLOCK_FILTERS = [64, 128, 256, 512]

    def __init__(self, model_path=None, is_training=False):
        if is_training:
            init_epoch, lr = self._get_training_info()
            logger.debug('training from epoch {}...'.format(init_epoch))
            model_path = self._model_file_path(init_epoch)
        if model_path is None or not os.path.isfile(model_path):
            self._construct_model()
        else:
            self._model = load_model(model_path, custom_objects={'loss': loss, 'acc_stone': acc_stone, 'acc_board': acc_board})

    def _construct_conv_layer(self, input, filters=64, kernel_size=(3, 3), strides=(1, 1)):
        output = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, data_format="channels_first", use_bias=False, padding='same', **self._REGULARIZERS)(input)
        output = BatchNormalization(axis=1)(output)
        output = Activation('relu')(output)
        return output

    def _construct_res_block(self, input, filters=64, kernel_size=(3, 3), reduce_dim=False):
        output = self._construct_conv_layer(input, filters, kernel_size, (2, 2) if reduce_dim else (1, 1))
        output = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), data_format="channels_first", use_bias=False, padding='same', **self._REGULARIZERS)(output)
        output = BatchNormalization(axis=1)(output)
        shortcut = input
        if reduce_dim:
            shortcut = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), data_format="channels_first", use_bias=False, padding='valid', **self._REGULARIZERS)(input)
        output = Add()([output, shortcut])
        output = Activation('relu')(output)
        return output

    def _construct_model(self):
        input = Input(shape=(IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
        output = self._construct_conv_layer(input, kernel_size=(7, 7), strides=(2, 2))    #112
        output = MaxPool2D(data_format="channels_first", padding='same')(output)          #56
        for i, block_count in enumerate(self._RES_BLOCK_COUNTS):                          #28->14->7
            for j in range(block_count):
                output = self._construct_res_block(output, filters=self._RES_BLOCK_FILTERS[i], reduce_dim=(i > 0 and j == 0))
        output = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), data_format="channels_first")(output)
        output = Flatten()(output)
        sm_list = []
        for i in range(BOARD_SIZE * BOARD_SIZE):
            out_i = Dense(3, activation='softmax', **self._REGULARIZERS)(output)
            sm_list.append(out_i)
        output = Concatenate()(sm_list)
        self._model = Model(inputs=input, outputs=output)
        self._model.compile(
            optimizer=optimizers.SGD(lr=LR, momentum=MOMENTUM),
            loss=loss,
            metrics=[acc_stone, acc_board]
        )

    def _callbacks(self):
        def lambdaCallbackFunc(epoch, _):
            print(K.eval(self._model.optimizer.lr))
            if (epoch + 1) % MODEL_SAVE_PERIOD == 0:
                self._save(epoch + 1)
                with open(self._model_config_file_path(), mode='w', encoding='utf-8') as f:
                    dic = {}
                    dic['epoch'] = epoch + 1
                    dic['lr'] = K.eval(self._model.optimizer.lr)
                    json.dump(dic, f, cls=NumpyEncoder)
        def learningRateSchedulerFunc(epoch):
            return LR * (0.5 ** (epoch // 8))
        return [
            # callbacks.ReduceLROnPlateau(monitor='loss', factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, epsilon=LR_REDUCE_EPSILON),
            # callbacks.ModelCheckpoint(os.path.join(proj_path, '{epoch:d}.hdf5'), period=MODEL_SAVE_PERIOD),
            # callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self._model.save_weights(os.path.join(model_dir, 'epoch_{}.hdf5'.format(epoch)))),
            callbacks.LearningRateScheduler(schedule=learningRateSchedulerFunc),
            callbacks.LambdaCallback(on_epoch_end=lambdaCallbackFunc),
            callbacks.TensorBoard(log_dir=self._model_dir(), batch_size=BATCH_SIZE)
        ]

    def train(self, images, boards):
        init_epoch, _ = self._get_training_info()
        self._model.fit(
            images,
            boards,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            initial_epoch=init_epoch,
            validation_split=VALIDATION_SPLIT,
            callbacks=self._callbacks()
        )

    def train_generator(self, generator, steps_per_epoch, val_generator=None, val_steps_per_epoch=None):
        init_epoch, _ = self._get_training_info()
        self._model.fit_generator(
            generator(True),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator(False) if val_generator is not None else val_generator,
            validation_steps=val_steps_per_epoch,
            epochs=EPOCHS,
            initial_epoch=init_epoch,
            workers=1,
            max_queue_size=10,
            callbacks=self._callbacks()
        )

    def predict(self, images):
        batch_size = np.shape(images)[0]
        output = np.zeros((batch_size, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        boards = self._model.predict(images)
        boards_cubic = np.reshape(boards, (batch_size, 3, BOARD_SIZE, BOARD_SIZE), order='F')
        indices = np.argmax(boards_cubic, axis=1)
        output[indices == 1] = -1
        output[indices == 0] = 1
        return output

    def evaluate(self, images, boards):
        return self._model.evaluate(images, boards, batch_size=BATCH_SIZE)

    def _save(self, epoch):
        filepath = self._model_file_path(epoch)
        # self._model.save_weights(filepath=filepath)
        self._model.save(filepath)

    @classmethod
    def _model_dir(cls):
        return os.path.join(model_dir, '{}_{}_{}_{}_{}_{}_{}'.format(
            TRAIN_IMAGE_SIZE,
            IMAGE_CHANNELS,
            *cls._RES_BLOCK_COUNTS,
            str(REG).replace('.', ''))
        )
        # return os.path.join(model_dir, '{}_{}_{}_{}_{}_{}'.format(
        #     TRAIN_IMAGE_SIZE,
        #     IMAGE_CHANNELS,
        #     *cls._RES_BLOCK_COUNTS,
        #     str(REG).replace('.', ''))
        # )

    @classmethod
    def _model_file_path(cls, epoch):
        return os.path.join(cls._model_dir(), '{}.hdf5'.format(epoch))

    @classmethod
    def _model_config_file_path(cls):
        return os.path.join(cls._model_dir(), 'config')

    @classmethod
    def _get_training_info(cls):
        epoch = 0
        lr = LR
        model_dir = cls._model_dir()
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if os.path.isfile(cls._model_config_file_path()):
            with open(cls._model_config_file_path(), mode='rt', encoding='utf-8') as f:
                dic = json.load(f)
                if dic['epoch'] is not None:
                    epoch = dic['epoch']
                if dic['lr'] is not None:
                    lr = dic['lr']
        return epoch, lr
