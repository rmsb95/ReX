import sys
import os
import pydicom
import numpy as np
from ReXNNPS import calculateNNPS
from ReXMTF import calculateMTF
from ReXDQE import calculateDQE
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from resources.main_window import Ui_MainWindow
from resources.dqe_window import Ui_Dialog as Form

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str)

    def __init__(self, taskType, path, functionType, a, b, exportFormat):
        super().__init__()
        self.taskType = taskType
        self.path = path
        self.functionType = functionType
        self.a = a
        self.b = b
        self.exportFormat = exportFormat

    def run(self):
        try:
            if self.taskType == "NNPS":
                self.log_signal.emit(">> -------------------------------")
                self.log_signal.emit(">> Cálculo de NPS y NNPS iniciado.")
                self.log_signal.emit(">> -------------------------------")
                calculateNNPS(self.path, self.functionType, self.a, self.b, self.exportFormat, self.progress.emit)
                self.log_signal.emit(">> ---------------------------------")
                self.log_signal.emit(">> Cálculo de NPS y NNPS finalizado.")
                self.log_signal.emit(">> ---------------------------------")

            elif self.taskType == "MTF":
                self.log_signal.emit(">> ------------------------")
                self.log_signal.emit(">> Cálculo de MTF iniciado.")
                self.log_signal.emit(">> ------------------------")
                calculateMTF(self.path, self.functionType, self.a, self.b, self.exportFormat, self.progress.emit)
                self.log_signal.emit(">> --------------------------")
                self.log_signal.emit(">> Cálculo de MTF finalizado.")
                self.log_signal.emit(">> --------------------------")

        except Exception as e:
            self.error.emit(str(e))

        finally:
            self.finished.emit()

class DQEWindow(QDialog, Form):
    log_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super(DQEWindow, self).__init__(parent)
        self.setupUi(self)
        self.setModal(True)
        self.nnps_file = None
        self.mtf_file = None
        self.calculateButton.clicked.connect(self.run_calculation)

        # Connect buttons with functions
        self.selectNPS.clicked.connect(self.select_nnps_file)
        self.selectMTF.clicked.connect(self.select_mtf_file)


    def select_nnps_file(self):
        # It opens a QFileDialog to select the NNPS_to_DQE file
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecciona el archivo NNPS_to_DQE", "", "All Files (*);;Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file_name:
            self.nnps_file = file_name
            print("Selected NNPS file:", file_name)

    def select_mtf_file(self):
        # It opens a QFileDialog to select the MTF_to_DQE file
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecciona el archivo MTF_to_DQE", "", "All Files (*);;Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file_name:
            self.mtf_file = file_name
            print("Selected MTF file:", file_name)

    def run_calculation(self):
        # Determinar el RQA seleccionado
        if self.Button_RQA3.isChecked():
            beamQuality = "RQA3"
        elif self.Button_RQA5.isChecked():
            beamQuality = "RQA5"
        elif self.Button_RQA7.isChecked():
            beamQuality = "RQA7"
        elif self.Button_RQA9.isChecked():
            beamQuality = "RQA9"

        kermaAire = self.kermaValue.value()

        # Comprobar que ambos archivos han sido seleccionados
        if not self.nnps_file or not self.mtf_file:
            QMessageBox.warning(self, "Error", "Por favor, selecciona ambos archivos antes de ejecutar el cálculo.")
            return

        # Llamada a la función que calcula la DQE
        result = calculateDQE(self.nnps_file, self.mtf_file, beamQuality, kermaAire)

        # Guardar el archivo
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Selecciona directorio para guardar el archivo", "/home/qt_user/Documents/DQE.xlsx",
                                                   "All Files (*);;Excel Files (*.xlsx);;CSV Files (*.csv)",
                                                   options=options)
        if file_path:
            if file_path.endswith('.csv'):
                result.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                result.to_excel(file_path, index=False)

        self.log_signal.emit("DQE calculada.")
        QMessageBox.information(self, "Resultado", f"La ejecución ha finalizado.")

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setupUi(self)

        self.setWindowIcon(QIcon('resources/dinosauricon.ico'))

        # Lista de archivos de imagen DICOM
        self.image_files = []
        self.current_index = -1
        self.directory= ""

        # Conectar botones a métodos
        self.prevButton.clicked.connect(self.show_previous_image)
        self.nextButton.clicked.connect(self.show_next_image)
        self.browseButton.clicked.connect(self.load_images_from_directory)
        self.Button_NPS.clicked.connect(self.execute_function_NPS)
        self.Button_MTF.clicked.connect(self.execute_function_MTF)
        self.Button_DQE.clicked.connect(self.open_dqe_window)

        # Inicialmente deshabilitar los botones de navegación
        self.prevButton.setEnabled(False)
        self.nextButton.setEnabled(False)

        # Inicializar el logText
        self.logText.setReadOnly(True)

        self.logText.append("Aplicación de ReX para el cálculo de la DQE iniciada.")
        self.logText.append(" >> Selecciona un directorio para iniciar el análisis.")

        # Conectar la señal log_signal al slot log_message
        self.dqe_window = DQEWindow(self)
        self.dqe_window.log_signal.connect(self.log_message)

    def open_dqe_window(self):
        self.dqe_window.exec()

    def log_message(self, message):
        self.logText.append(message)

    def load_images_from_directory(self):
        # Abre un cuadro de diálogo para seleccionar el directorio
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio", os.path.expanduser("~"))
        if directory:
            self.directory = directory
            # Lista de archivos en el directorio
            self.image_files = []
            self.logText.append(f"Has seleccionado el directorio: {directory}")
            for f in os.listdir(directory):
                file_path = os.path.join(directory, f)
                try:
                    dicom_image = pydicom.dcmread(file_path, force=True)
                    # Asegurarse de que el archivo tiene un pixel_array válido
                    if hasattr(dicom_image, 'pixel_array'):
                        self.image_files.append(file_path)
                        print(f"Archivo DICOM válido {file_path}")
                        self.logText.append(f"Archivo DICOM válido: {file_path}")
                except Exception as e:
                    print(f"Archivo no válido {file_path}: {e}")
                    #self.logText.append(f"Archivo no válido {file_path}: {e}")

            self.image_files.sort()  # Ordenar archivos para una navegación coherente
            if self.image_files:
                self.current_index = 0
                self.show_image(self.current_index)
                # Habilitar los botones de navegación si hay más de una imagen
                self.update_navigation_buttons()
                self.logText.append(">> Imágenes cargadas. Ajusta los parámetros y selecciona una función.")
            else:
                QMessageBox.warning(self, "Advertencia",
                                    "No se encontraron imágenes DICOM válidas en el directorio seleccionado.")
                self.prevButton.setEnabled(False)
                self.nextButton.setEnabled(False)

    def apply_window(self, image_array):
        lower = np.min(image_array)
        upper = np.max(image_array)

        # Escala los píxeles dentro del rango de la ventana a 0-65535 (para Grayscale16)
        windowed_image = (np.clip(image_array, lower, upper) - lower) / (upper - lower) * 65535
        windowed_image = windowed_image.astype(np.uint16)

        return windowed_image

    def show_image(self, index):
        if 0 <= index < len(self.image_files):
            dicom_data = pydicom.dcmread(self.image_files[index])
            image_array = dicom_data.pixel_array

            # Apliquemos un ajuste de ventana (ajusta estos valores según tus necesidades)
            windowed_image = self.apply_window(image_array)

            # Crear un objeto QImage desde el array de numpy
            bytes_per_line = windowed_image.shape[1] * 2
            q_image = QImage(windowed_image, windowed_image.shape[1], windowed_image.shape[0], bytes_per_line, QImage.Format_Grayscale16)

            # Copiar datos para asegurar la persistencia fuera del ámbito de esta función
            q_image = q_image.copy()

            # Convertir QImage a QPixmap y mostrarlo en el label
            self.imageLabel.setPixmap(QPixmap.fromImage(q_image))
            self.imageLabel.setScaledContents(True)

            # Actualizar el nombre de la imagen
            self.imageNameLabel.setText(os.path.basename(self.image_files[index]))

            self.current_image_index = index
        else:
            QMessageBox.information(self, "Error", "No image available.")

    def show_previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.current_index)
            self.update_navigation_buttons()

    def show_next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image(self.current_index)
            self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.prevButton.setEnabled(self.current_index > 0)
        self.nextButton.setEnabled(self.current_index < len(self.image_files) - 1)

    def execute_function_NPS(self):
        # Obtener el estado de los radio buttons
        if self.radioButton1.isChecked():
            functionType = "linear"
        elif self.radioButton2.isChecked():
            functionType = "log"
        else:
            functionType = "linear"

        if self.radioButton3.isChecked():
            exportFormat = "excel"
        elif self.radioButton4.isChecked():
            exportFormat = "csv"
        else:
            exportFormat = "excel"

        a = self.doubleSpinBox_a.value()
        b = self.doubleSpinBox_b.value()
        path = self.directory

        if not path:
            QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún directorio.")
            return

        if a == 0:
            QMessageBox.warning(self, "Advertencia", "El parámetro a no puede ser igual 0.")
            return

        # Cambiar el cursor a cursor de espera
        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Crear y configurar el trabajador
        self.worker = Worker('NNPS', path, functionType, a, b, exportFormat)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.show_error)
        self.worker.log_signal.connect(self.log_message)

        # Deshabilitar el botón mientras se ejecuta la tarea
        self.Button_NPS.setEnabled(False)
        self.Button_MTF.setEnabled(False)
        self.Button_DQE.setEnabled(False)

        # Iniciar la tarea en un hilo separado
        self.worker.start()

    def update_progress(self, value):
        self.progressBar.setValue(value)

    def on_finished(self):
        # Restaurar el cursor a normal
        QApplication.restoreOverrideCursor()

        # Habilitar el botón y mostrar mensaje de finalización
        self.Button_NPS.setEnabled(True)
        self.Button_MTF.setEnabled(True)
        self.Button_DQE.setEnabled(True)
        QMessageBox.information(self, "Completado", "El cálculo ha finalizado correctamente.")

    def show_error(self, error_message):
        # Devolver la barra de progreso a 0
        self.progressBar.setValue(0)

        # Restaurar el cursor a normal:
        QApplication.restoreOverrideCursor()

        # Mostrar mensaje de error y habilitar el botón
        self.Button_NPS.setEnabled(True)
        self.Button_MTF.setEnabled(True)
        self.Button_DQE.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Se produjo un error: {error_message}")

    def execute_function_MTF(self):
        if self.radioButton1.isChecked():
            functionType = "linear"
        elif self.radioButton2.isChecked():
            functionType = "log"
        else:
            functionType = "linear"

        if self.radioButton3.isChecked():
            exportFormat = "excel"
        elif self.radioButton4.isChecked():
            exportFormat = "csv"
        else:
            exportFormat = "excel"

        a = self.doubleSpinBox_a.value()
        b = self.doubleSpinBox_b.value()
        path = self.directory

        if not path:
            QMessageBox.warning(self, "Advertencia", "No se ha seleccionado ningún directorio.")
            return

        if a == 0:
            QMessageBox.warning(self, "Advertencia", "El parámetro a no puede ser igual 0.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.worker = Worker('MTF', path, functionType, a, b, exportFormat)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.show_error)

        self.Button_MTF.setEnabled(False)
        self.Button_NPS.setEnabled(False)
        self.Button_DQE.setEnabled(False)

        self.worker.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
