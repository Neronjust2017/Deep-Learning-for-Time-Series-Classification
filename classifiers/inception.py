# resnet model
import keras
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics
from utils.utils import save_test_duration


class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True, build=True, batch_size=16,   # 默认64
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=200):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        # [40,20,10]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)

        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        file_path_val = self.output_directory + 'best_model_val.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        model_checkpoint_val = keras.callbacks.ModelCheckpoint(filepath=file_path_val, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint,model_checkpoint_val]

        model.summary()

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=True):
        if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # 将测试集划分为val和test
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
        print("test label:")
        print(y_test)

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        # y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
        #                       return_df_metrics=False)
        y_pred = self.predict(x_test, y_true, return_df_metrics=False,Is_val=False)

        y_pred_val = self.predict(x_test,y_true, return_df_metrics=False,Is_val=True)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        np.save(self.output_directory + 'y_pred_val.npy', y_pred_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        y_pred_val = np.argmax(y_pred_val,axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_pred_val, y_true, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, return_df_metrics=True,Is_val=False):
        start_time = time.time()
        if Is_val:
            model_path = self.output_directory + 'best_model_val.hdf5'
        else:
            model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            if Is_val:
                save_test_duration(self.output_directory + 'test_duration_val.csv', test_duration)
            else:
                save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
