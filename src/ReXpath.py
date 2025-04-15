# ------------------------------------------------------ #
#                 DICOM Manager System                   #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 06-2024
# Version: 1.0

import sys
import os
import pydicom
import shutil

import src.ReXfunc as ReX
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QMessageBox, QListWidgetItem

# Cargar la interfaz gráfica generada por PyQt Designer
from src.UI.rexpath_window import Ui_Dialog

dino_icon_path = ReX.resource_path("resources/dinosauricon.ico")
tagline_path = ReX.resource_path('resources/HUVM-tagline.png')

class DICOMOrganizer(QMainWindow, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon(dino_icon_path))
        self.selected_tags = []

        # Conectar los botones a sus respectivas funciones
        self.load_dicom_btn.clicked.connect(self.load_dicom_file)
        self.select_dir_btn.clicked.connect(self.select_directory)
        self.process_btn.clicked.connect(self.process_directory)

        # Configurar la tabla
        self.dicom_table.setColumnCount(3)
        self.dicom_table.setHorizontalHeaderLabels(['Tag', 'Name', 'Value'])
        self.dicom_table.setSelectionBehavior(self.dicom_table.SelectRows)
        self.dicom_table.setSelectionMode(self.dicom_table.MultiSelection)
        self.dicom_table.itemSelectionChanged.connect(self.update_selected_tags)

    def load_dicom_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Selecciona un archivo DICOM", "", "DICOM Files (*.dcm);;All Files (*)",
                                                   options=options)
        if file_name:
            self.read_dicom_tags(file_name)

    def read_dicom_tags(self, file_name):
        self.dicom_table.setRowCount(0)
        ds = pydicom.dcmread(file_name)
        for elem in ds:
            if elem.VR != 'SQ':
                row_position = self.dicom_table.rowCount()
                self.dicom_table.insertRow(row_position)
                tag_item = QTableWidgetItem(str(elem.tag))
                tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsEditable)
                self.dicom_table.setItem(row_position, 0, tag_item)

                name_item = QTableWidgetItem(elem.name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                self.dicom_table.setItem(row_position, 1, name_item)

                value_item = QTableWidgetItem(str(elem.value))
                value_item.setFlags(value_item.flags() & ~Qt.ItemIsEditable)
                self.dicom_table.setItem(row_position, 2, value_item)

    # def update_selected_tags(self):
    #     self.selected_tags_list.clear()
    #     self.selected_tags = []
    #     selected_rows = self.dicom_table.selectionModel().selectedRows()
    #     for row in selected_rows:
    #         tag_str = self.dicom_table.item(row.row(), 0).text()
    #         tag_tuple = tuple(int(t, 16) for t in tag_str.strip("()").split(", "))
    #         tag_name = self.dicom_table.item(row.row(), 1).text()
    #         self.selected_tags.append((tag_tuple, tag_name))
    #     for tag in self.selected_tags:
    #         item = QListWidgetItem(f"{tag[1]} ({tag[0]})")
    #         self.selected_tags_list.addItem(item)

    def update_selected_tags(self):
        self.selected_tags_list.clear()
        self.selected_tags = []
        selected_rows = self.dicom_table.selectionModel().selectedRows()
        for row in selected_rows:
            tag_str = self.dicom_table.item(row.row(), 0).text()
            group, element = tag_str.strip("()").split(",")
            # Mantener los valores originales como enteros
            group_val = int(group.strip(), 16)
            element_val = int(element.strip(), 16)
            tag_tuple = (group_val, element_val)
            tag_name = self.dicom_table.item(row.row(), 1).text()
            self.selected_tags.append((tag_tuple, tag_name))
        for tag, name in self.selected_tags:
            # Formatear cada número hexadecimal por separado
            formatted_tag = f"({tag[0]:04x},{tag[1]:04x})"
            item = QListWidgetItem(f"{name} {formatted_tag}")
            self.selected_tags_list.addItem(item)

    def select_directory(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Selecciona la ruta para ordenar los archivos", options=options)
        if directory:
            self.directory = directory

    def process_directory(self):
        if not hasattr(self, 'directory'):
            QMessageBox.warning(self, 'Advertencia', 'Por favor, selecciona primero un directorio.')
            return
        if not self.selected_tags:
            QMessageBox.warning(self, 'Advertencia', 'Por favor, selecciona al menos un tag para ordenar.')
            return

        self.organize_dicom_files(self.directory)
        QMessageBox.information(self, '¡Éxito!', 'Los archivos se han ordenado.')

    def organize_dicom_files(self, directory):
        for subdir, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(subdir, file)
                # if not file.lower().endswith('.dcm'):
                #     #self.report_invalid_file(file_path, 'Invalid DICOM file')
                #     continue
                if not ReX.is_dicom_file(file_path):
                    self.report_invalid_file(file_path, 'Invalid DICOM file')
                    continue

                try:
                    ds = pydicom.dcmread(file_path)
                except:
                    self.report_invalid_file(file_path, 'Unable to read DICOM file')
                    continue

                dest_dir = self.create_destination_directory(ds)
                if not dest_dir:
                    dest_dir = os.path.join(directory, 'Unsorted')
                os.makedirs(dest_dir, exist_ok=True)

                shutil.copy(file_path, dest_dir)

    def create_destination_directory(self, ds):
        dir_parts = []
        for tag, name in self.selected_tags:
            element = ds.get(tag)
            if element is None:
                print(f"Tag {tag} no encontrado.")
                return None
            try:
                value = element.value
                dir_parts.append(f"{name}_{str(value)}")
                print(f"Tag: {tag}, Name: {name}, Value: {value}")
            except Exception as e:
                print(f"Error retrieving value for tag {tag}: {e}")
                return None
        return os.path.join(self.directory, *dir_parts)

    def report_invalid_file(self, file_path, message):
        print(f"Error con el archivo {file_path}: {message}")
        QMessageBox.warning(self, 'Warning', f"Error con el archivo {file_path}: {message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DICOMOrganizer()
    ex.show()
    sys.exit(app.exec_())
