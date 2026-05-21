# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ReX_DQE.ui'
#
# Created by: PyQt5 UI code generator 5.15.10 (modified manually)
#
# Modifications:
#   - RQA radio buttons replaced by a QComboBox (beamQualityCombo).
#   - Added snr2Label to display the SNR² value for the selected beam quality.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(420, 330)

        # --- Beam quality selector ---
        self.RQALabel = QtWidgets.QLabel(Dialog)
        self.RQALabel.setGeometry(QtCore.QRect(20, 20, 261, 20))
        self.RQALabel.setObjectName("RQALabel")

        self.beamQualityCombo = QtWidgets.QComboBox(Dialog)
        self.beamQualityCombo.setGeometry(QtCore.QRect(20, 45, 371, 30))
        self.beamQualityCombo.setObjectName("beamQualityCombo")

        # SNR² info label — updated dynamically when the combo selection changes
        self.snr2Label = QtWidgets.QLabel(Dialog)
        self.snr2Label.setGeometry(QtCore.QRect(20, 80, 371, 20))
        self.snr2Label.setObjectName("snr2Label")
        self.snr2Label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        # --- Air kerma input ---
        self.kermaLabel = QtWidgets.QLabel(Dialog)
        self.kermaLabel.setGeometry(QtCore.QRect(20, 120, 261, 20))
        self.kermaLabel.setObjectName("kermaLabel")

        self.kermaValue = QtWidgets.QDoubleSpinBox(Dialog)
        self.kermaValue.setGeometry(QtCore.QRect(290, 113, 101, 31))
        self.kermaValue.setMaximum(99999.99)
        self.kermaValue.setDecimals(3)
        self.kermaValue.setProperty("value", 2.5)
        self.kermaValue.setObjectName("kermaValue")

        # --- MTF file selector ---
        self.MTFLabel = QtWidgets.QLabel(Dialog)
        self.MTFLabel.setGeometry(QtCore.QRect(20, 170, 261, 30))
        self.MTFLabel.setObjectName("MTFLabel")

        self.selectMTF = QtWidgets.QPushButton(Dialog)
        self.selectMTF.setGeometry(QtCore.QRect(290, 170, 101, 31))
        self.selectMTF.setObjectName("selectMTF")

        # --- NNPS file selector ---
        self.NPSLabel = QtWidgets.QLabel(Dialog)
        self.NPSLabel.setGeometry(QtCore.QRect(20, 220, 261, 30))
        self.NPSLabel.setObjectName("NPSLabel")

        self.selectNPS = QtWidgets.QPushButton(Dialog)
        self.selectNPS.setGeometry(QtCore.QRect(290, 220, 101, 31))
        self.selectNPS.setObjectName("selectNPS")

        # --- Calculate button ---
        self.calculateButton = QtWidgets.QPushButton(Dialog)
        self.calculateButton.setGeometry(QtCore.QRect(155, 280, 110, 30))
        self.calculateButton.setObjectName("calculateButton")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ReX - DQE"))

        self.RQALabel.setText(_translate("Dialog", "Selecciona la calidad del haz empleado:"))
        self.kermaLabel.setText(_translate("Dialog", "Introduce el Kerma en aire en μGy:"))
        self.MTFLabel.setText(_translate("Dialog", "Selecciona el archivo MTF_to_DQE:"))
        self.selectMTF.setText(_translate("Dialog", "Archivo MTF"))
        self.NPSLabel.setText(_translate("Dialog", "Selecciona el archivo NNPS_to_DQE:"))
        self.selectNPS.setText(_translate("Dialog", "Archivo NNPS"))
        self.calculateButton.setText(_translate("Dialog", "Calcular DQE"))

        # Populate beam quality combo box.
        # Keys must match exactly those in ReXDQE.SNR2_TABLE.
        beam_qualities = [
            "RQA3",
            "RQA5",
            "RQA7",
            "RQA9",
            "RQA-M1 Mo/Mo",
            "RQA-M2 Mo/Mo",
            "RQA-M3 Mo/Mo",
            "RQA-M4 Mo/Mo",
            "Mo/Rh 28kV",
            "Rh/Rh 28kV",
            "W/Rh 28kV",
            "W/Al 28kV",
            'W/Rh 29kV'
        ]
        for quality in beam_qualities:
            self.beamQualityCombo.addItem(_translate("Dialog", quality))

        # Default selection: RQA5 (index 1), consistent with previous default.
        self.beamQualityCombo.setCurrentIndex(1)

        # snr2Label initial text is set by DQEWindow.update_snr2_label()
        # after the signal is connected; leave it empty here.
        self.snr2Label.setText("")