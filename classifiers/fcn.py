# FCN
import keras 
import numpy as np 
import pandas as pd 
import time 
from keras.callbacks import  TensorBoard
# 如果是 from tensorflow.keras.callbacks import  TensorBoard 会报错： Model' object has no attribute 'run_eagerly'
from utils.utils import save_logs
from utils.metric import precision,recall,f1

class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
		self.output_directory = output_directory
		self.model = self.build_model(input_shape, nb_classes)
		if(verbose==True):
			self.model.summary()
		self.verbose = verbose
		self.model.save_weights(self.output_directory+'model_init.hdf5')

	def build_model(self, input_shape, nb_classes):

		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.normalization.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.normalization.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(filters=128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.normalization.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

		model = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(), 
			metrics=['accuracy',precision,recall,f1])

		# reduce learning rate    min_lr :学习率下限
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
			min_lr=0.0001)

		file_path = self.output_directory+'best_model.hdf5'

		file_path_val = self.output_directory + 'best_model_val.hdf5'

		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
		 	save_best_only=True)

		# val_loss 检测
		model_checkpoint_val = keras.callbacks.ModelCheckpoint(filepath=file_path_val, monitor='val_loss',
														   save_best_only=True)

		# tensorboard 进行可视化

		tbCallBack = TensorBoard(log_dir='model',histogram_freq=1,write_grads=True)

		self.callbacks = [reduce_lr,model_checkpoint,model_checkpoint_val,tbCallBack]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true,nb_classes):
		# x_val and y_val are only used to monitor the test loss and NOT for training  
		batch_size = 16
		nb_epochs = 100

		# 将测试集划分为val和test
		print(x_val.shape[0] / 2)

		l = int(x_val.shape[0]/2)
		x_test = x_val[l:]
		y_test = y_val[l:]
		x_val = x_val[:l]
		y_val = y_val[:l]


		y_true = y_true[int(y_true.shape[0]/2):]

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



		mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

		start_time = time.time() 

		# hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
							  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
		
		duration = time.time() - start_time

		model = keras.models.load_model(self.output_directory+'best_model.hdf5',custom_objects={'precision': precision,'recall':recall,'f1':f1})

		model_val=keras.models.load_model(self.output_directory+'best_model_val.hdf5',custom_objects={'precision': precision,'recall':recall,'f1':f1})

		# y_pred = model.predict(x_val)
		# y_pred_val = model_val.predict(x_val)

		y_pred = model.predict(x_test)
		y_pred_val = model_val.predict(x_test)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)
		y_pred_val = np.argmax(y_pred_val,axis=1)

		save_logs(self.output_directory, hist, y_pred, y_pred_val, y_true, duration,nb_classes)

		keras.backend.clear_session()

	
