# ------------------------------------------------------ #
#                           REX                          #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 04-2025
# Version: 1.0.1

import sys
import os
import openpyxl
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import src.ReXfunc as ReX
from src.ReXNNPS import calculateNNPS
from src.ReXMTF import calculateMTF
from src.ReXDQE import calculateDQE
from src.ReXpath import DICOMOrganizer
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from src.UI.main_window import Ui_MainWindow
from src.UI.dqe_window import Ui_Dialog as Form


# Rutas de iconos
dino_icon_path = ReX.resource_path("resources/dinosauricon.ico")
tagline_path = ReX.resource_path('resources/HUVM-tagline.png')

class Worker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool)
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
        self.success = False

    def run(self):
        try:
            if self.taskType == "NNPS":
                self.log_signal.emit(">> Cálculo de NPS y NNPS iniciado.")
                self.results = calculateNNPS(self.path, self.functionType, self.a, self.b, self.exportFormat, self.progress.emit)

            elif self.taskType == "MTF":
                self.log_signal.emit(">> Cálculo de MTF iniciado.")
                self.results = calculateMTF(self.path, self.functionType, self.a, self.b, self.exportFormat, self.progress.emit)

            # Una ejecución correcta implica:
            self.success = True

        except Exception as e:
            self.error.emit(str(e))

            # Una ejecución incorrecta implica:
            self.success = False

        finally:
            self.finished.emit(self.success)

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
            if 'NNPS_to_DQE' in file_name:
                self.nnps_file = file_name
                print("Selected NNPS file:", file_name)
            else:
                QMessageBox.warning(self, "Advertencia", "El archivo seleccionado no es NNPS_to_DQE.")
                self.nnps_file = None

    def select_mtf_file(self):
        # It opens a QFileDialog to select the MTF_to_DQE file
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecciona el archivo MTF_to_DQE", "", "All Files (*);;Excel Files (*.xlsx);;CSV Files (*.csv)")
        if file_name:
            if 'MTF_to_DQE' in file_name:
                self.mtf_file = file_name
                print("Selected MTF file:", file_name)
            else:
                QMessageBox.warning(self, "Advertencia", "El archivo seleccionado no es MTF_to_DQE.")
                self.mtf_file = None

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

        # Mostrar resultados en tabla y gráfica # PENDIENTE REVISAR
        # ReX.show_results_table_and_graph(result, "Resultados de DQE")

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

        self.setWindowIcon(QIcon(dino_icon_path))

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
        self.actionOrganizador_DICOM.triggered.connect(self.open_rexpath_window)
        self.actionHelp.triggered.connect(self.show_help)
        self.actionAbout_us.triggered.connect(self.show_about)


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

    def open_rexpath_window(self):
        self.rexpath_window = DICOMOrganizer()
        self.rexpath_window.show()

    def show_help(self):
        help_text = """
        Manual de Usuario - ReX

        1. Carga de Imágenes:
           - Haga clic en 'Examinar' para seleccionar un directorio con imágenes DICOM.
             
             > Requisitos para cálculo de MTF:
             - Debe haber como mucho dos imágenes DICOM en el directorio para el cálculo de MTF,
               y ninguna imagen de otro tipo.
             - Si hay una imagen, puede ser en vertical o en horizontal.
             - Si hay dos imágenes, una debe ser en vertical y la otra en horizontal.
             - Para el posterior cálculo de la DQE, debe haber dos imágenes.
             
             > Requisitos para cálculo de NPS y NNPS:
             - Debe haber al menos una imagen DICOM en el directorio para el cálculo de NPS,
               y ninguna imagen de otro tipo.               
               
           - Use los botones de navegación para ver las imágenes y comprobar los requisitos.

        2. Cálculo de NPS y NNPS:
           - Seleccione el tipo de función (lineal/logarítmica).
           - Ajuste los parámetros a y b.
           - Elija el formato de exportación: csv o excel.
           - Haga clic en 'Calcular NNPS'.

        3. Cálculo de MTF:
           - Siga los mismos pasos que para NNPS. Recuerde seleccionar el directorio adecuado.
           - Haga clic en 'Calcular MTF'.

        4. Cálculo de DQE:
           - Es necesario haber calculado antes la MTF y el NNPS.
           - Seleccione archivos NNPS_to_DQE y MTF_to_DQE generados en los cálculos previos. 
           - Elija la calidad del haz.
           - Introduzca el valor de kerma.
           - Haga clic en 'Calcular DQE'.
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Ayuda de ReX")
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

    def show_about(self):
        about_text = """
        ReX - Herramienta de Análisis de Calidad de Imagen

        Versión: 2025.1
        Desarrolladores: 
        - Rafael Manuel Segovia Brome (@rmsb95)
        - Antonio Ortiz Lora (@aol)

        Hospital Universitario Virgen Macarena
        Servicio de Radiofísica
        Sevilla, España

        © 2025
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Acerca de ReX")
        msg.setText(about_text)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()

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

    def on_finished(self, success):
        # Restaurar el cursor a normal
        QApplication.restoreOverrideCursor()

        # Habilitar el botón y mostrar mensaje de finalización
        self.Button_NPS.setEnabled(True)
        self.Button_MTF.setEnabled(True)
        self.Button_DQE.setEnabled(True)

        if success:
            QMessageBox.information(self, "Completado", "El cálculo ha finalizado correctamente.")
            # Mostrar resultados en tabla y gráfica
            if hasattr(self.worker, 'results'):
                ReX.show_results_table_and_graph(self.worker.results, f"Resultados de {self.worker.taskType}")


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
        self.worker.log_signal.connect(self.log_message)

        self.Button_MTF.setEnabled(False)
        self.Button_NPS.setEnabled(False)
        self.Button_DQE.setEnabled(False)

        self.worker.start()

class SplashScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ReX")
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Fondo de la pantalla de inicio
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1f22;
                border: 2px solid #2b952b;
                border-radius: 10px;
            }
        """)

        # Layout y etiquetas
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("ReX")
        title.setStyleSheet("font-size: 48px; font-weight: bold; color: white;")
        layout.addWidget(title)

        subtitle = QLabel("Desarrolladores: @rmsb95 @aol\nVersión: 2024.1")
        subtitle.setStyleSheet("font-size: 14px; color: white;")
        layout.addWidget(subtitle)

        # Espacio para la imagen
        image_label = QLabel()
        pixmap = QPixmap(tagline_path)
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        layout.addWidget(image_label)

        self.setLayout(layout)

        # Temporizador para cerrar la pantalla de inicio y abrir la ventana principal
        QTimer.singleShot(5000, self.show_main_window)  # Cerrar después de 5 segundos

    def show_main_window(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

def main():
    app = QApplication(sys.argv)

    # Mostrar la pantalla de inicio
    splash = SplashScreen()
    splash.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
