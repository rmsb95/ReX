# ------------------------------------------------------ #
#                 ReX auxiliar functions                 #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 05-2024
# Version: 1.0

import os
import openpyxl
import numpy as np
import pandas as pd
import math
import pydicom
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QApplication, QAction
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from skimage import data, filters
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline



def resource_path(relative_path):
    # Si está empaquetado, usar sys._MEIPASS
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        # Ruta relativa desde el directorio raíz (no src/)
        base_path = os.path.abspath(os.path.dirname(__file__) + '/../')
    return os.path.join(base_path, relative_path)


dino_icon_path = resource_path("resources/dinosauricon.ico")

def cropImage(image, roiSizeX, roiSizeY, delta_x, delta_y, despCenterX, despCenterY):
    """
    Obtiene un ROI central a partir de la imagen proporcionada.
    Convierte las medidas a píxeles y calcula el centro desplazado.

    Parámetros:
    - image: matriz de la imagen de entrada.
    - roi_size_x, roi_size_y: tamaño del ROI en unidades de medida.
    - delta_x, delta_y: tamaño del pixel en las mismas unidades que las dimensiones del ROI.
    - desp_center_x, desp_center_y: displacement del centro del ROI en píxeles.

    Retorna:
    - cropped_image: imagen recortada.
    """
    # Convertir las medidas a píxeles
    cropWidth = round(roiSizeX / delta_x)
    cropHeight = round(roiSizeY / delta_y)

    # Calcular el centro de la imagen
    height, width = image.shape[:2]              # COMPROBAR QUE ESTO FUNCIONA
    centerX = round(width / 2) + despCenterX
    centerY = round(height / 2) + despCenterY

    # Definir los límites del recorte
    startX = round(max(centerX - cropWidth // 2, 0))
    startY = round(max(centerY - cropHeight // 2, 0))
    endX = startX + cropWidth                    # SE ELIMINA EL -1 RESPECTO A MATLAB
    endY = startY + cropHeight                   # SE ELIMINA EL -1 RESPECTO A MATLAB

    # Asegurar que no excedemos los límites de la imagen
    endX = min(endX, width)
    endY = min(endY, height)

    # Recortar la imagen
    croppedImage = image[startY:endY, startX:endX]
    return croppedImage, cropHeight, cropWidth


def evaluateCentering(image, meanDose):
    """
    Evaluates if there are pixels in an image with a dose significantly lower than the mean dose.

    Parameters:
    image (numpy array): A 2D array representing the image where each element is the dose at a pixel.
    meanDose (float): The mean dose calculated from the image.

    Returns:
    bool: True if there is at least one pixel with a dose less than or equal to 80% of the mean dose, False otherwise.
    """
    # Calculate the threshold as 80% of the mean dose
    threshold = 0.8 * meanDose

    # Create a boolean mask where each element is True if the corresponding pixel in the image
    # has a value less than or equal to the threshold, and False otherwise
    lowerPixels = image <= threshold

    # Check if there is at least one pixel with a True value in the mask
    areThereLowerPixels = np.any(lowerPixels)

    return areThereLowerPixels

def adjustedImage(imageData):
    """
    Adjusts the image data by fitting a polynomial surface and subtracting it to obtain residuals.

    Parameters:
    imageData (numpy array): A 2D array representing the image data.

    Returns:
    numpy array: A 2D array representing the residual image after polynomial adjustment.
    """
    # Create a mesh according to image dimensions
    X, Y = np.meshgrid(np.arange(imageData.shape[1]), np.arange(imageData.shape[0]))

    # Transform image data type to float
    Z = imageData.astype(float)

    # Define polynomial function
    def poly22(XY, a, b, c, d, e, f):
        x, y = XY
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    # Flatten the arrays for the fitting function
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = Z.ravel()

    # Polynomial adjusting
    popt, pcov = curve_fit(poly22, xdata, ydata)

    # Calculate surface from polynomial adjusting
    # Ensure X and Y are passed as separate arrays within a single array
    S = poly22((X, Y), *popt)

    # Subtract the polynomial from the original image to obtain the residuals
    residualImage = Z - S

    return residualImage


def correctedImage(imageData):
    """
    Corrects the image data by fitting a polynomial surface and adjusting the values to normalize the image.

    Parameters:
    imageData (numpy array): A 2D array representing the image data.

    Returns:
    numpy array: A 2D array representing the corrected image.
    """
    # Create a mesh according to image dimensions
    X, Y = np.meshgrid(np.arange(imageData.shape[1]), np.arange(imageData.shape[0]))

    # Transform image data type to float
    Z = imageData.astype(float)

    # Define polynomial function
    def poly22(XY, a, b, c, d, e, f):
        x, y = XY
        return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    # Flatten the arrays for the fitting function
    xdata = np.vstack((X.ravel(), Y.ravel()))
    ydata = Z.ravel()

    # Polynomial adjusting
    popt, pcov = curve_fit(poly22, xdata, ydata)

    # Calculate surface from polynomial adjusting
    # Ensure X and Y are passed as separate arrays within a single array
    S = poly22((X, Y), *popt)

    # Average value of the best fitted surface
    avgS = np.mean(S)

    # Apply correction
    corrImage = Z / S * avgS

    return corrImage

def edgeDetection(image):
    """
    Detects edges in the image and determines the orientation of the edges using PCA.

    Parameters:
    image (numpy array): A 2D array representing the image data.

    Returns:
    tuple: A tuple containing the angle of the first principal component and the string orientation.
    """
    # Apply the Roberts filter to detect edges in the image
    edges = filters.roberts(image)

    # Threshold the edge-detected image to obtain a binary matrix
    # This value may require adjustment
    threshold = 0.5
    edgeMatrix = (edges > threshold).astype(int)

    # plt.figure()  # Creates a new figure window
    # plt.imshow(edgeMatrix, cmap='gray')  # Display the original image
    # plt.colorbar()  # Display a color bar
    # plt.title('Edge Matrix')  # Title of the figure
    # plt.show()

    # Extract coordinates of the edge points
    y, x = np.where(edgeMatrix == 1)
    points = np.column_stack((x, y))

    # Apply PCA to the edge points
    pca = PCA(n_components=2)
    pca.fit(points)

    # The first principal component represents the direction of maximum variance
    first_component = pca.components_[0]
    angle_rad = np.arctan2(first_component[1], first_component[0])
    angle = angle_rad * 180 / math.pi

    # Determine string orientation based on the angle
    absangle = abs(angle)
    if 0 <= absangle <= 5:
        orientation = 'horizontal'
    elif 85 <= absangle <= 95:
        orientation = 'vertical'
    else:
        orientation = 'error'

    return angle, orientation

def exportData(X, Y1, Y2, col_names, path, name, format):
    """
    Exports data to a specified format (Excel or CSV).

    Parameters:
    X (array-like): The data for the first column.
    Y1 (array-like): The data for the second column.
    Y2 (array-like): The data for the third column.
    col_names (list): A list of column names for the DataFrame.
    path (str): The directory path where the file will be saved.
    name (str): The name of the file to be saved.
    format (str): The format of the file to be saved ('excel' or 'csv').

    Raises:
    ValueError: If the format is not supported.
    """
    # Check if Y2 is different from Y1
    if np.array_equal(Y1, Y2):
        Y2 = np.zeros(len(Y2))

    # Create a dataframe with the data
    data = pd.DataFrame({
        col_names[0]: X,
        col_names[1]: Y1,
        col_names[2]: Y2
    })

    # Just in case, covnert format to lowercase
    format = format.lower()

    # Ensure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Verify format and save the file accordingly
    if format == 'excel':
        # Prepare the full path
        fullname = path + '/' + name + '.xlsx'
        filename = os.path.join(path, fullname)
        # Write the dataframe to an Excel file
        data.to_excel(filename, index=False)
        print(f'Data saved in Excel file: {filename}')

    elif format == 'csv':
        # Prepare the full path
        fullname = path + '/' + name + '.csv'
        filename = os.path.join(path, fullname)
        # Write the dataframe to a CSV file
        data.to_csv(filename, index=False)
        print(f'Data saved in CSV file: {filename}')

    else:
        raise ValueError('Format not supported. Use "excel" o "csv".')

def read_data(file_path, exportFormat):
    """ Read data from a file in Excel or CSV format."""
    if exportFormat == 'excel':
        return pd.read_excel(file_path)
    elif exportFormat == 'csv':
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsopported file format for {file_path}")

def find_closest(df, target_frequency, frequencyColumnName):
    """Find the row with the closest frequency to the target frequency."""
    idx = (df[frequencyColumnName] - target_frequency).abs().idxmin()
    return df.loc[idx]

def fit_polynomial_and_predict(df, target_frequencies, frequencyColumnName, valuesColumnName):
    """Fit a third-degree polynomial to the data and predict values for target frequencies"""
    # Remove NaN values from the relevant columns
    df = df[[frequencyColumnName, valuesColumnName]].dropna()

    # In case of NNPS, we eliminate first values for better fit
    if valuesColumnName in ['NNPS Horizontal', 'NNPS Vertical']:
        df = df[df[frequencyColumnName] >= 0.5]

    # Extract frequencies and values
    frequencies = df[frequencyColumnName].values
    values = df[valuesColumnName].values

    # Fit a third-degree polynomial
    poly_coeffs = np.polyfit(frequencies, values, 3)
    print(poly_coeffs)

    # Predict values for the target frequencies
    predict_values = np.polyval(poly_coeffs, target_frequencies)

    return predict_values

def  fit_spline_and_predict(df, target_frequencies, frequencyColumnName, valuesColumnName):
    """Fit a cubic spline to the data and predict values for target frequencies"""
    # Remove NaN values from the relevant columns
    df = df[[frequencyColumnName, valuesColumnName]].dropna()

    # Extract frequencies and values
    frequencies = df[frequencyColumnName].values
    values = df[valuesColumnName].values

    # Fit a cubic spline
    spline = CubicSpline(frequencies, values)

    # Predict values for the target frequencies
    predict_values = spline(target_frequencies)

    return predict_values

def process_file(file_path, target_frequencies, valuesName, exportFormat):
    """Process a file to find the predicted MTF or NNPS values for the target frequencies using polynomial fitting."""
    try:
        # Read data from the file
        df = read_data(file_path, exportFormat)
        df = df.dropna(axis=1, how='all')

        # Fit polynomial and predict values
        predicted_values = fit_spline_and_predict(df, target_frequencies, 'Frequencies (1/mm)', valuesName)
        return dict(zip(target_frequencies, predicted_values))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None


def is_dicom_file(filepath):
    """Check if a file is a valid DICOM file"""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception as e:
        return False

def find_dicom_files(directory):
    """Find all DICOM files in a directory, regardless of their file extension"""
    dicom_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)
    return dicom_files

def show_results_table_and_graph(results, title):
    show_results_graph(results, title)
    class TableDialog(QDialog):
        def __init__(self, results, title, parent=None):
            super().__init__(parent)
            self.setWindowTitle(title)
            self.setWindowIcon(QIcon(dino_icon_path))
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)  # Eliminar el icono de ayuda
            self.resize(425, 450)

            layout = QVBoxLayout()
            self.table = QTableWidget()
            df = pd.DataFrame(results)
            df.reset_index(inplace=True)  # Restablecer el índice para incluir las frecuencias como columna

            # Formatear solo las columnas de NNPS a notación científica con dos decimales
            for column in df.columns:
                if column != 'Frequencies (1/mm)':
                    df[column] = df[column].apply(lambda x: f'{x:.2e}')

            self.table.setRowCount(df.shape[0])
            self.table.setColumnCount(df.shape[1])
            self.table.setHorizontalHeaderLabels(df.columns)

            for i in range(df.shape[0]):
                for j in range(df.shape[1]):
                    item = QTableWidgetItem(str(df.iat[i, j]))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Hacer la celda de solo lectura
                    self.table.setItem(i, j, item)

            self.table.setSelectionBehavior(QTableWidget.SelectItems)
            self.table.setSelectionMode(QTableWidget.ExtendedSelection)
            self.table.setContextMenuPolicy(Qt.ActionsContextMenu)

            copy_action = QAction("Copiar", self)
            copy_action.setShortcut(QKeySequence.Copy)
            copy_action.triggered.connect(self.copy_selection)
            self.addAction(copy_action)

            layout.addWidget(self.table)
            self.setLayout(layout)

        def copy_selection(self):
            selection = self.table.selectedIndexes()
            if selection:
                # Obtener filas y columnas únicas seleccionadas
                rows = sorted(set(index.row() for index in selection))
                columns = sorted(set(index.column() for index in selection))
                rowcount = rows[-1] - rows[0] + 1
                colcount = columns[-1] - columns[0] + 1

                # Crear una tabla con el tamaño adecuado para incluir los encabezados
                table = [[''] * colcount for _ in range(rowcount)]

                # Añadir los encabezados de las columnas solo una vez
                headers = [self.table.horizontalHeaderItem(col).text() for col in columns]
                table.insert(0, headers)  # Insertar los encabezados en la primera fila

                # Llenar la tabla con los datos seleccionados
                for index in selection:
                    row = index.row() - rows[0] + 1  # Ajustar para los encabezados
                    col = index.column() - columns[0]
                    table[row][col] = self.table.item(index.row(), index.column()).text()

                # Convertir la tabla a una cadena de texto tabulada
                stream = '\n'.join('\t'.join(row) for row in table)
                clipboard = QApplication.clipboard()
                clipboard.setText(stream)

    dialog = TableDialog(results, title)
    dialog.exec_()

def show_results_graph(results, title):
    df = pd.DataFrame(results)

    # Create plot
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.set_xlabel('Frecuencias (1/mm)')
    ax.set_ylabel('Valor')
    ax.set_title(title)
    ax.legend()

    manager = plt.get_current_fig_manager()
    manager.set_window_title(title)
    fig.canvas.manager.window.setWindowIcon(QIcon('resources/dinosauricon.ico'))

    plt.show()