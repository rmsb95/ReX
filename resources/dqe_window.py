# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ReX_DQE.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(468, 368)
        self.RQALabel = QtWidgets.QLabel(Dialog)
        self.RQALabel.setGeometry(QtCore.QRect(30, 30, 341, 16))
        self.RQALabel.setObjectName("RQALabel")
        self.kermaLabel = QtWidgets.QLabel(Dialog)
        self.kermaLabel.setGeometry(QtCore.QRect(30, 140, 341, 16))
        self.kermaLabel.setObjectName("kermaLabel")
        self.kermaValue = QtWidgets.QDoubleSpinBox(Dialog)
        self.kermaValue.setGeometry(QtCore.QRect(260, 130, 101, 31))
        self.kermaValue.setMaximum(99999.99)
        self.kermaValue.setProperty("value", 2.5)
        self.kermaValue.setObjectName("kermaValue")
        self.calculateButton = QtWidgets.QPushButton(Dialog)
        self.calculateButton.setGeometry(QtCore.QRect(180, 310, 93, 28))
        self.calculateButton.setObjectName("calculateButton")
        self.MTFLabel = QtWidgets.QLabel(Dialog)
        self.MTFLabel.setGeometry(QtCore.QRect(30, 190, 221, 31))
        self.MTFLabel.setObjectName("MTFLabel")
        self.selectMTF = QtWidgets.QPushButton(Dialog)
        self.selectMTF.setGeometry(QtCore.QRect(260, 190, 101, 31))
        self.selectMTF.setObjectName("selectMTF")
        self.NPSLabel = QtWidgets.QLabel(Dialog)
        self.NPSLabel.setGeometry(QtCore.QRect(30, 250, 221, 31))
        self.NPSLabel.setObjectName("NPSLabel")
        self.selectNPS = QtWidgets.QPushButton(Dialog)
        self.selectNPS.setGeometry(QtCore.QRect(260, 250, 101, 31))
        self.selectNPS.setObjectName("selectNPS")
        self.horizontalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(100, 60, 272, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Button_RQA3 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.Button_RQA3.setObjectName("Button_RQA3")
        self.horizontalLayout.addWidget(self.Button_RQA3)
        self.Button_RQA5 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.Button_RQA5.setChecked(True)
        self.Button_RQA5.setObjectName("Button_RQA5")
        self.horizontalLayout.addWidget(self.Button_RQA5)
        self.Button_RQA7 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.Button_RQA7.setObjectName("Button_RQA7")
        self.horizontalLayout.addWidget(self.Button_RQA7)
        self.Button_RQA9 = QtWidgets.QRadioButton(self.horizontalLayoutWidget)
        self.Button_RQA9.setObjectName("Button_RQA9")
        self.horizontalLayout.addWidget(self.Button_RQA9)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ReX - DQE"))
        self.RQALabel.setText(_translate("Dialog", "Selecciona la calidad del haz empleado:"))
        self.kermaLabel.setText(_translate("Dialog", "Introduce el Kerma en aire en μGy:"))
        self.calculateButton.setText(_translate("Dialog", "Calcular DQE"))
        self.MTFLabel.setText(_translate("Dialog", "Selecciona el archivo MTF_to_DQE:"))
        self.selectMTF.setText(_translate("Dialog", "Archivo MTF"))
        self.NPSLabel.setText(_translate("Dialog", "Selecciona el archivo NNPS_to_DQE:"))
        self.selectNPS.setText(_translate("Dialog", "Archivo NNPS"))
        self.Button_RQA3.setText(_translate("Dialog", "RQA 3"))
        self.Button_RQA5.setText(_translate("Dialog", "RQA 5"))
        self.Button_RQA7.setText(_translate("Dialog", "RQA 7"))
        self.Button_RQA9.setText(_translate("Dialog", "RQA 9"))
