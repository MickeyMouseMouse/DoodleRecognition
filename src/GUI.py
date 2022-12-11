import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
import threading
import subprocess
import webbrowser
from ModelTraining import ModelTraining


class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(600, 250)
		MainWindow.setMinimumSize(QtCore.QSize(600, 250))
		MainWindow.setMaximumSize(QtCore.QSize(600, 250))
		font = QtGui.QFont()
		font.setPointSize(12)
		MainWindow.setFont(font)
		icon = QtGui.QIcon()
		icon.addPixmap(QtGui.QPixmap("../resources/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
		MainWindow.setWindowIcon(icon)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setMinimumSize(QtCore.QSize(600, 250))
		self.centralwidget.setMaximumSize(QtCore.QSize(600, 250))
		self.centralwidget.setObjectName("centralwidget")
		self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
		self.tabWidget.setGeometry(QtCore.QRect(0, 0, 600, 250))
		self.tabWidget.setMinimumSize(QtCore.QSize(600, 250))
		self.tabWidget.setMaximumSize(QtCore.QSize(600, 250))
		self.tabWidget.setObjectName("tabWidget")
		self.trainingTab = QtWidgets.QWidget()
		self.trainingTab.setObjectName("trainingTab")
		self.label = QtWidgets.QLabel(self.trainingTab)
		self.label.setGeometry(QtCore.QRect(10, 15, 111, 17))
		self.label.setObjectName("label")
		self.label_2 = QtWidgets.QLabel(self.trainingTab)
		self.label_2.setGeometry(QtCore.QRect(10, 75, 80, 17))
		self.label_2.setObjectName("label_2")
		self.label_3 = QtWidgets.QLabel(self.trainingTab)
		self.label_3.setGeometry(QtCore.QRect(10, 115, 80, 17))
		self.label_3.setObjectName("label_3")
		self.label_4 = QtWidgets.QLabel(self.trainingTab)
		self.label_4.setGeometry(QtCore.QRect(10, 155, 80, 17))
		self.label_4.setObjectName("label_4")
		self.datasetPath = QtWidgets.QLineEdit(self.trainingTab)
		self.datasetPath.setGeometry(QtCore.QRect(130, 10, 421, 30))
		self.datasetPath.setObjectName("datasetPath")
		self.btnDatasetBrowse = QtWidgets.QPushButton(self.trainingTab)
		self.btnDatasetBrowse.setGeometry(QtCore.QRect(560, 10, 31, 30))
		self.btnDatasetBrowse.setObjectName("btnDatasetBrowse")
		self.epochs = QtWidgets.QLineEdit(self.trainingTab)
		self.epochs.setGeometry(QtCore.QRect(100, 70, 40, 30))
		self.epochs.setMaxLength(2)
		self.epochs.setAlignment(QtCore.Qt.AlignCenter)
		self.epochs.setObjectName("epochs")
		self.optimizer = QtWidgets.QComboBox(self.trainingTab)
		self.optimizer.setGeometry(QtCore.QRect(100, 110, 100, 30))
		self.optimizer.addItem("SGD")
		self.optimizer.addItem("Adam")
		self.optimizer.addItem("AdamW")
		self.optimizer.addItem("Adadelta")
		self.optimizer.addItem("Adagrad")
		self.optimizer.addItem("Adamax")
		self.optimizer.addItem("Adafactor")
		self.optimizer.addItem("Nadam")
		self.optimizer.addItem("Ftrl")
		self.optimizer.setCurrentIndex(1)
		self.optimizer.setObjectName("optimizer")
		self.loss = QtWidgets.QComboBox(self.trainingTab)
		self.loss.setGeometry(QtCore.QRect(100, 150, 270, 30))
		self.loss.addItem("sparse_categorical_crossentropy")
		self.loss.addItem("poisson")
		self.loss.addItem("kl_divergence")
		self.loss.setCurrentIndex(0)
		self.loss.setObjectName("loss")
		self.btnTrain = QtWidgets.QPushButton(self.trainingTab)
		self.btnTrain.setGeometry(QtCore.QRect(501, 170, 90, 40))
		self.btnTrain.setObjectName("btnTrain")
		self.tabWidget.addTab(self.trainingTab, "")
		self.testingTab = QtWidgets.QWidget()
		self.testingTab.setObjectName("testingTab")
		self.label_5 = QtWidgets.QLabel(self.testingTab)
		self.label_5.setGeometry(QtCore.QRect(10, 15, 81, 17))
		self.label_5.setObjectName("label_5")
		self.modelPath = QtWidgets.QLineEdit(self.testingTab)
		self.modelPath.setGeometry(QtCore.QRect(100, 10, 451, 30))
		self.modelPath.setObjectName("modelPath")
		self.btnModelBrowse = QtWidgets.QPushButton(self.testingTab)
		self.btnModelBrowse.setGeometry(QtCore.QRect(560, 10, 31, 30))
		self.btnModelBrowse.setObjectName("btnModelBrowse")
		self.label_6 = QtWidgets.QLabel(self.testingTab)
		self.label_6.setGeometry(QtCore.QRect(10, 75, 121, 17))
		self.label_6.setObjectName("label_6")
		self.port = QtWidgets.QLineEdit(self.testingTab)
		self.port.setGeometry(QtCore.QRect(140, 70, 71, 30))
		self.port.setMaxLength(5)
		self.port.setAlignment(QtCore.Qt.AlignCenter)
		self.port.setObjectName("port")
		self.btnStart = QtWidgets.QPushButton(self.testingTab)
		self.btnStart.setGeometry(QtCore.QRect(501, 170, 90, 40))
		self.btnStart.setObjectName("btnStart")
		self.btnStop = QtWidgets.QPushButton(self.testingTab)
		self.btnStop.setEnabled(False)
		self.btnStop.setGeometry(QtCore.QRect(400, 170, 90, 40))
		self.btnStop.setObjectName("btnStop")
		self.tabWidget.addTab(self.testingTab, "")
		MainWindow.setCentralWidget(self.centralwidget)

		self.btnDatasetBrowse.clicked.connect(self.openDirectoryChooser)
		self.btnTrain.clicked.connect(self.startTraining)
		
		self.btnModelBrowse.clicked.connect(self.openFileChooser)
		self.btnStart.clicked.connect(self.startWebServer)
		self.btnStop.clicked.connect(self.stopWebServer)

		self.retranslateUi(MainWindow)
		self.tabWidget.setCurrentIndex(0)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Neural Network"))
		self.label.setText(_translate("MainWindow", "Dataset folder:"))
		self.label_2.setText(_translate("MainWindow", "Epochs:"))
		self.label_3.setText(_translate("MainWindow", "Optimizer:"))
		self.label_4.setText(_translate("MainWindow", "Loss:"))
		self.btnDatasetBrowse.setText(_translate("MainWindow", "..."))
		self.epochs.setText(_translate("MainWindow", "14"))
		self.btnTrain.setText(_translate("MainWindow", "Train"))
		self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainingTab), _translate("MainWindow", "Training"))
		self.label_5.setText(_translate("MainWindow", "Model file:"))
		defaultModelPath = Path().cwd() / Path("./static/models/model.tflite")
		if defaultModelPath.exists():
			self.modelPath.setText(str(defaultModelPath.resolve()))
		self.btnModelBrowse.setText(_translate("MainWindow", "..."))
		self.label_6.setText(_translate("MainWindow", "Web Server Port:"))
		self.port.setText(_translate("MainWindow", "8000"))
		self.btnStart.setText(_translate("MainWindow", "Start"))
		self.btnStop.setText(_translate("MainWindow", "Stop"))
		self.tabWidget.setTabText(self.tabWidget.indexOf(self.testingTab), _translate("MainWindow", "Testing"))
	
	def openDirectoryChooser(self):
		dataset_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")
		if dataset_path:
			self.datasetPath.setText(dataset_path)
		
	def startTraining(self):
		dataset_path = self.datasetPath.text()
		if len(dataset_path) == 0 or not Path(dataset_path).is_dir():
			QtWidgets.QMessageBox.critical(None, "Error", "Dataset path is invalid")
			return
		
		epochs = self.epochs.text()
		if not epochs.isdigit() or int(epochs) < 1:
			QtWidgets.QMessageBox.critical(None, "Error", "Epochs number is invalid")
			return
		
		self.btnTrain.setEnabled(False)
		
		global modelTraining
		modelTraining = ModelTraining(dataset_path, epochs, self.optimizer.currentText(), self.loss.currentText())
		modelTraining.signaller.finished.connect(self.onTrainingFinished)
		modelTraining.start()
	
	def onTrainingFinished(self):
		global modelTraining
		modelTraining.join()
		del modelTraining
		
		self.btnTrain.setEnabled(True)
		self.modelPath.setText(str((Path.cwd() / Path("./static/models/model.tflite")).resolve()))
		QtWidgets.QMessageBox.about(None, "Success", "The model is trained")
	
	def openFileChooser(self):
		model_path = QtWidgets.QFileDialog.getOpenFileName(None, "Open file", "", "TFLite (*.tflite)")[0]
		if model_path:
			self.modelPath.setText(model_path)

	def startWebServer(self):
		model_path = self.modelPath.text()
		if len(model_path) == 0 or not Path(model_path).is_file():
			QtWidgets.QMessageBox.critical(None, "Error", "Model path is invalid")
			return
		
		port = self.port.text()
		if not port.isdigit() or int(port) < 1024 or int(port) > 65535:
			QtWidgets.QMessageBox.critical(None, "Error", "Port number is invalid")
			return
		
		self.btnStart.setEnabled(False)
		self.btnStop.setEnabled(True)
		
		global webServerProcess
		webServerProcess = subprocess.Popen(["gunicorn", "-b", "127.0.0.1:" + port,
			"-w", "4", "-k", "uvicorn.workers.UvicornWorker", "WebServer:app"])
		webbrowser.open("http://localhost:" + port, new = 2)
	
	def stopWebServer(self):
		global webServerProcess
		webServerProcess.terminate()
		
		self.btnStart.setEnabled(True)
		self.btnStop.setEnabled(False)


if __name__ == "__main__":
	import sys
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())
