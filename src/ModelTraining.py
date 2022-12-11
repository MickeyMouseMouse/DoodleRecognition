from PyQt5 import QtCore
import threading
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization


class Signaller(QtCore.QObject):
	finished = QtCore.pyqtSignal()


class ModelTraining(threading.Thread):
	def __init__(self, dataset_path, epochs, optimizer, loss):
		super().__init__(daemon = True)
		self.dataset_path = dataset_path
		self.epochs = int(epochs)
		self.optimizer = optimizer
		self.loss = loss
		self.signaller = Signaller()
	
	def run(self):		
		image_size = (32, 32)
		batch_size = 32
		train_ds = image_dataset_from_directory(
			self.dataset_path,
			validation_split=0.2,
			subset="training",
			seed=123,
			image_size=image_size,
			batch_size=batch_size
		)
		val_ds = image_dataset_from_directory(
			self.dataset_path,
			validation_split=0.2,
			subset="validation",
			seed=123,
			image_size=image_size,
			batch_size=batch_size
		)
		
		labels = []
		for (image, label) in tuple(train_ds.unbatch()):
			labels.append(label.numpy())
		labels = pd.Series(labels)
		n_classes = len(labels.value_counts())
		
		input_shape = (32, 32, 3)
		
		model = Sequential([
			Rescaling(1. / 255, input_shape=input_shape),
			BatchNormalization(),

			Conv2D(6, kernel_size = (3, 3), padding = "same", activation = "relu"),
			Conv2D(8, kernel_size = (3, 3), padding = "same", activation = "relu"),
			Conv2D(10, kernel_size = (3, 3), padding = "same", activation = "relu"),
			BatchNormalization(),
			MaxPooling2D(pool_size = (2, 2)),

			Flatten(),

			Dense(900, activation = "relu"),
			BatchNormalization(),
			Dropout(0.1),

			Dense(500, activation = "relu"),
			BatchNormalization(),
			Dropout(0.1),

			Dense(400, activation = "relu"),
			Dropout(0.1),

			Dense(n_classes, activation = "softmax")
		])
		
		model.compile(
			optimizer = self.optimizer,
			loss = self.loss,
			metrics = ["accuracy"]
		)
		
		model.fit(
			train_ds,
			validation_data = val_ds,
			epochs = self.epochs,
			verbose = 1
		)
		
		converter = tf.lite.TFLiteConverter.from_keras_model(model)
		tflite_model = converter.convert()
		
		with open("./static/models/model.tflite", "wb") as f:
			f.write(tflite_model)
		
		self.signaller.finished.emit()


