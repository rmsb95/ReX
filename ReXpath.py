import sys
import os
import pydicom
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTableWidget, \
    QTableWidgetItem, QListWidget, QMessageBox
from PyQt5.QtCore import Qt


class DICOMOrganizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selected_tags = []

    def initUI(self):
        self.setWindowTitle('DICOM Organizer')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.dicom_table = QTableWidget()
        self.dicom_table.setColumnCount(3)
        self.dicom_table.setHorizontalHeaderLabels(['Tag', 'Name', 'Value'])
        self.dicom_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dicom_table.setSelectionMode(QTableWidget.MultiSelection)
        layout.addWidget(self.dicom_table)

        self.selected_tags_list = QListWidget()
        layout.addWidget(self.selected_tags_list)

        btn_layout = QHBoxLayout()
        load_dicom_btn = QPushButton('Load DICOM')
        load_dicom_btn.clicked.connect(self.load_dicom_file)
        btn_layout.addWidget(load_dicom_btn)

        select_dir_btn = QPushButton('Select Directory')
        select_dir_btn.clicked.connect(self.select_directory)
        btn_layout.addWidget(select_dir_btn)

        process_btn = QPushButton('Process')
        process_btn.clicked.connect(self.process_directory)
        btn_layout.addWidget(process_btn)

        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def load_dicom_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select DICOM File", "", "DICOM Files (*.dcm);;All Files (*)",
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
                self.dicom_table.setItem(row_position, 0, QTableWidgetItem(str(elem.tag)))
                self.dicom_table.setItem(row_position, 1, QTableWidgetItem(str(elem.name)))
                self.dicom_table.setItem(row_position, 2, QTableWidgetItem(str(elem.value)))

        self.dicom_table.itemSelectionChanged.connect(self.update_selected_tags)

    def update_selected_tags(self):
        self.selected_tags_list.clear()
        self.selected_tags = []
        for item in self.dicom_table.selectedItems():
            if item.column() == 0:
                tag_str = item.text()
                tag_tuple = tuple(int(t, 16) for t in tag_str.strip("()").split(", "))
                self.selected_tags.append(tag_tuple)
        self.selected_tags_list.addItems([str(tag) for tag in self.selected_tags])

    def select_directory(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if directory:
            self.directory = directory

    def process_directory(self):
        if not hasattr(self, 'directory'):
            QMessageBox.warning(self, 'Warning', 'Please select a directory first.')
            return
        if not self.selected_tags:
            QMessageBox.warning(self, 'Warning', 'Please select at least one tag.')
            return

        self.organize_dicom_files(self.directory)

    def organize_dicom_files(self, directory):
        for subdir, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(subdir, file)
                if not file.lower().endswith('.dcm'):
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
        for tag in self.selected_tags:
            element = ds.get(tag)  # Utilizamos ds.get(tag) en lugar de ds.data_element(tag)
            if element is None:
                print(f"Tag {tag} not found")
                return None
            try:
                value = element.value
                dir_parts.append(str(value))
                print(f"Tag: {tag}, Value: {value}")
            except Exception as e:
                print(f"Error retrieving value for tag {tag}: {e}")
                return None
        return os.path.join(self.directory, *dir_parts)

    def report_invalid_file(self, file_path, message):
        print(f"Error with file {file_path}: {message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DICOMOrganizer()
    ex.show()
    sys.exit(app.exec_())
