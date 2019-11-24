import keras
import  numpy as np
import time
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras import Input
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Activation, \
    BatchNormalization, Conv1D, MaxPooling1D, Concatenate, Lambda, Reshape, UpSampling1D
from utils.utils import save_logs
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.resnet_v2.loss import my_loss, f1
from utils.resnet_v2.tools import AdvancedLearnignRateScheduler

class Classifier_RESNET_V2:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,type=1):
        self.output_directory = output_directory
        if type==1:
            self.model = self.build_model(input_shape, nb_classes)
        elif type==2:
            self.model = self.build_model_without_dropout(input_shape,nb_classes)
        elif type==3:
            self.model = self.build_model_with_L2(input_shape,nb_classes)

        if (verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self,input_shape,nb_classes):
        OUTPUT_CLASS = nb_classes
        input1 = Input(shape=input_shape, name='input_ecg')
        k = 1  # increment every 4th residual block
        p = True  # pool toggle every other residual block (end with 2^8)
        convfilt = 64
        encoder_confilt = 64  # encoder filters' num
        convstr = 1
        ksize = 16
        poolsize = 2
        poolstr = 2
        drop = 0.5

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr,
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
        lcount += 1
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
        # Left branch (convolutions)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x1)
        # Right branch, shortcut branch pooling
        x2 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x)
        # Merge both branches
        x = keras.layers.add([x1, x2])
        del x1, x2

        fms = []
        ## Main loop
        p = not p
        for l in range(15):

            if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
                k += 1
                # increase depth by 1x1 Convolution case dimension shall change
                xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
                lcount += 1
            else:
                xshort = x
                # Left branch (convolutions)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr,
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = BatchNormalization(name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr,
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            if p:
                x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

                # Right branch: shortcut connection
            if p:
                x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
            else:
                x2 = xshort  # pool or identity
            # Merging branches
            x = keras.layers.add([x1, x2])
            # change parameters
            p = not p  # toggle pooling
            # if l == 5:
            #     fms.append(x)
            # if l == 6:
            #     fms.append(x)
            #     fms.append(x)
            #     fms.append(x)
        # fms的内容：[<tf.Tensor 'add_6/Identity:0' shape=(None, 1136, 128) dtype=float32>,
        #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>,
        #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>,
        #             <tf.Tensor 'add_7/Identity:0' shape=(None, 1136, 128) dtype=float32>]

        # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
        # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

        # Final bit
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        x_ecg = Flatten()(x)

        # bbox_num = 1
        #
        # x2od2 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[0])
        # out2 = Reshape((1136, bbox_num, 2), name='aux_output1')(x2od2)
        #
        # x2od3 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[1])
        # out3 = Reshape((1136, bbox_num, 2), name='aux_output2')(x2od3)
        #
        # x2od4 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[2])
        # out4 = Reshape((1136, bbox_num, 2), name='aux_output3')(x2od4)
        #
        # x2od5 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[3])
        # out5 = Reshape((1136, bbox_num, 2), name='aux_output4')(x2od5)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        # model = Model(inputs=input1, outputs=[out1, out2, out3, out4, out5])
        model = Model(inputs=input1, outputs=out1)

        adam = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])
        file_path = self.output_directory + 'best_model.hdf5'
        file_path_val = self.output_directory + 'best_model_val.hdf5'
        log_path = self.output_directory + 'log.csv'

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, verbose=1),
            AdvancedLearnignRateScheduler(monitor='val_loss', patience=3, verbose=1, mode='auto',
                                          decayRatio=0.1),
            ModelCheckpoint(filepath=file_path_val,
                            monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint(filepath=file_path,
                            monitor='loss',save_best_only=True,verbose=1),
            keras.callbacks.CSVLogger(log_path, separator=',',
                                      append=True)
        ]


        self.callbacks = callbacks

        model.summary()

        return model

    def build_model_without_dropout(self,input_shape,nb_classes):
        OUTPUT_CLASS = nb_classes  # output classes

        input1 = Input(shape=input_shape, name='input_ecg')
        k = 1  # increment every 4th residual block
        p = True  # pool toggle every other residual block (end with 2^8)
        convfilt = 64
        encoder_confilt = 64  # encoder filters' num
        convstr = 1
        ksize = 16
        poolsize = 2
        poolstr = 2

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr,
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
        lcount += 1
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
        # Left branch (convolutions)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr,
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x1)
        # Right branch, shortcut branch pooling
        x2 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x)
        # Merge both branches
        x = keras.layers.add([x1, x2])
        del x1, x2

        fms = []
        ## Main loop
        p = not p
        for l in range(15):

            if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
                k += 1
                # increase depth by 1x1 Convolution case dimension shall change
                xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
                lcount += 1
            else:
                xshort = x
                # Left branch (convolutions)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr,
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = BatchNormalization(name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr,
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            if p:
                x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

                # Right branch: shortcut connection
            if p:
                x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
            else:
                x2 = xshort  # pool or identity
            # Merging branches
            x = keras.layers.add([x1, x2])
            # change parameters
            p = not p  # toggle pooling
            # if l == 5:
            #     fms.append(x)
            # if l == 6:
            #     fms.append(x)
            #     fms.append(x)
            #     fms.append(x)

        # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
        # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

        # Final bit
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        x_ecg = Flatten()(x)

        # bbox_num = 1
        #
        # x2od2 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[0])
        # out2 = Reshape((1136, bbox_num, 2), name='aux_output1')(x2od2)
        #
        # x2od3 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[1])
        # out3 = Reshape((1136, bbox_num, 2), name='aux_output2')(x2od3)
        #
        # x2od4 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[2])
        # out4 = Reshape((1136, bbox_num, 2), name='aux_output3')(x2od4)
        #
        # x2od5 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[3])
        # out5 = Reshape((1136, bbox_num, 2), name='aux_output4')(x2od5)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output')(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        adam = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])
        file_path = self.output_directory + 'best_model.hdf5'
        file_path_val = self.output_directory + 'best_model_val.hdf5'
        log_path = self.output_directory + 'log.csv'

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            AdvancedLearnignRateScheduler(monitor='val_loss', patience=3, verbose=1, mode='auto',
                                          decayRatio=0.1),
            ModelCheckpoint(filepath=file_path_val,
                            monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint(file_path=file_path,
                            monitor='loss', save_best_only=True, verbose=1),
            keras.callbacks.CSVLogger(log_path, separator=',',
                                      append=True)
        ]

        self.callbacks = callbacks

        model.summary()

        return model

    def build_model_with_L2(self,input_shape,nb_classes):

        OUTPUT_CLASS = nb_classes # output classes

        input1 = Input(shape=input_shape, name='input_ecg')
        k = 1  # increment every 4th residual block
        p = True  # pool toggle every other residual block (end with 2^8)
        convfilt = 64
        encoder_confilt = 64  # encoder filters' num
        convstr = 1
        ksize = 16
        poolsize = 2
        poolstr = 2
        drop = 0.5

        # First convolutional block (conv,BN, relu)
        lcount = 0
        x = Conv1D(filters=convfilt,
                   kernel_size=ksize,
                   padding='same',
                   strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                   kernel_initializer='he_normal', name='layer' + str(lcount))(input1)
        lcount += 1
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        ## Second convolutional block (conv, BN, relu, dropout, conv) with residual net
        # Left branch (convolutions)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x)
        lcount += 1
        x1 = BatchNormalization(name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = Activation('relu')(x1)
        x1 = Dropout(drop)(x1)
        x1 = Conv1D(filters=convfilt,
                    kernel_size=ksize,
                    padding='same',
                    strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                    kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
        lcount += 1
        x1 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x1)
        # Right branch, shortcut branch pooling
        x2 = MaxPooling1D(pool_size=poolsize,
                          strides=poolstr, padding='same')(x)
        # Merge both branches
        x = keras.layers.add([x1, x2])
        del x1, x2

        fms = []
        ## Main loop
        p = not p
        for l in range(15):

            if (l % 4 == 0) and (l > 0):  # increment k on every fourth residual block
                k += 1
                # increase depth by 1x1 Convolution case dimension shall change
                xshort = Conv1D(filters=convfilt * k, kernel_size=1, name='layer' + str(lcount))(x)
                lcount += 1
            else:
                xshort = x
                # Left branch (convolutions)
            # notice the ordering of the operations has changed
            x1 = BatchNormalization(name='layer' + str(lcount))(x)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = BatchNormalization(name='layer' + str(lcount))(x1)
            lcount += 1
            x1 = Activation('relu')(x1)
            x1 = Dropout(drop)(x1)
            x1 = Conv1D(filters=convfilt * k,
                        kernel_size=ksize,
                        padding='same',
                        strides=convstr, kernel_regularizer=keras.regularizers.l2(0.001),
                        kernel_initializer='he_normal', name='layer' + str(lcount))(x1)
            lcount += 1
            if p:
                x1 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(x1)

                # Right branch: shortcut connection
            if p:
                x2 = MaxPooling1D(pool_size=poolsize, strides=poolstr, padding='same')(xshort)
            else:
                x2 = xshort  # pool or identity
            # Merging branches
            x = keras.layers.add([x1, x2])
            # change parameters
            p = not p  # toggle pooling
            # if l == 5:
            #     fms.append(x)
            # if l == 6:
            #     fms.append(x)
            #     fms.append(x)
            #     fms.append(x)

        # x = Conv1D(filters=convfilt * k, kernel_size=ksize, padding='same', strides=convstr, kernel_initializer='he_normal')(x)
        # x_reg = Conv1D(filters=convfilt * k, kernel_size=1, padding='same', strides=convstr, kernel_initializer='he_normal')(x)

        # Final bit
        x = BatchNormalization(name='layer' + str(lcount))(x)
        lcount += 1
        x = Activation('relu')(x)

        x_ecg = Flatten()(x)

        # bbox_num = 1
        #
        # x2od2 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[0])
        # out2 = Reshape((1136, bbox_num, 2), name='aux_output1')(x2od2)
        #
        # x2od3 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[1])
        # out3 = Reshape((1136, bbox_num, 2), name='aux_output2')(x2od3)
        #
        # x2od4 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[2])
        # out4 = Reshape((1136, bbox_num, 2), name='aux_output3')(x2od4)
        #
        # x2od5 = Conv1D(filters=bbox_num * 2, kernel_size=1, padding='same', strides=convstr,
        #                kernel_initializer='he_normal')(
        #     fms[3])
        # out5 = Reshape((1136, bbox_num, 2), name='aux_output4')(x2od5)

        out1 = Dense(OUTPUT_CLASS, activation='softmax', name='main_output',
                     kernel_regularizer=keras.regularizers.l2(0.001), )(x_ecg)

        model = Model(inputs=input1, outputs=out1)

        adam = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam,
                      metrics=['accuracy'])
        file_path = self.output_directory + 'best_model.hdf5'
        file_path_val = self.output_directory + 'best_model_val.hdf5'
        log_path = self.output_directory + 'log.csv'

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=1),
            AdvancedLearnignRateScheduler(monitor='val_loss', patience=3, verbose=1, mode='auto',
                                          decayRatio=0.1),
            ModelCheckpoint(filepath=file_path_val,
                            monitor='val_loss', save_best_only=True, verbose=1),
            ModelCheckpoint(file_path=file_path,
                            monitor='loss', save_best_only=True, verbose=1),
            keras.callbacks.CSVLogger(log_path, separator=',',
                                      append=True)
        ]

        self.callbacks = callbacks

        model.summary()

        return model


    def fit(self, x_train, y_train, x_val, y_val, y_true):
        batch_size = 16
        nb_epochs = 100
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        print(x_val.shape[0] / 2)

        l = int(x_val.shape[0] / 2)
        x_test = x_val[l:]
        y_test = y_val[l:]
        x_val = x_val[:l]
        y_val = y_val[:l]

        y_true = y_true[int(y_true.shape[0] / 2):]

        print("train:")
        print(x_train)
        print("train label:")
        print(y_train)
        print("val:")
        print(x_val)
        print("val label:")
        print(y_val)
        print("test:")
        print(x_test)
        print("train label:")
        print(y_test)

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory+'best_model.hdf5')
        model_val = keras.models.load_model(self.output_directory + 'best_model_val.hdf5')

        y_pred = model.predict(x_test)
        y_pred_val = model_val.predict(x_test)

        y_pred = np.argmax(y_pred, axis=1)
        y_pred_val = np.argmax(y_pred_val, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_pred_val, y_true, duration,lr=False)

        keras.backend.clear_session()











