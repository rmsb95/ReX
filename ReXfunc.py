import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from skimage import data, filters
from sklearn.decomposition import PCA


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
    threshold = 0.8 * meanDose
    lowerPixels = image <= threshold
    areThereLowerPixels = np.any(lowerPixels)

    # Show figure for visual evaluation
    #plt.figure()
    #plt.imshow(lowerPixels, cmap='gray')
    #plt.title('Pixel values below threshold')
    #plt.colorbar()
    #plt.show()

    return areThereLowerPixels

def adjustedImage(imageData):
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

    # Subtract the polynomial from the original image to obtain the residuals.
    residualImage = Z - S

    return residualImage


def correctedImage(imageData):
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
    # TBR: APPLY DIFFERENT EDGE_ALGORITHM
    # Create a boolean matrix where 1 represents the edge
    # edgeMatrix = filters.sobel(image)

    # Aplicar el filtro de Sobel para detectar bordes en la imagen
    edges = filters.roberts(image)

    # Umbralizar la imagen para obtener una matriz binaria
    # Este umbral puede requerir ajustes según la imagen
    threshold = 0.5  # Este es un valor de ejemplo, puede necesitar ser ajustado
    edgeMatrix = (edges > threshold).astype(int)

    # plt.figure()  # Creates a new figure window
    # plt.imshow(edgeMatrix, cmap='gray')  # Display the original image
    # plt.colorbar()  # Display a color bar
    # plt.title('Edge Matrix')  # Title of the figure
    # plt.show()

    # Extraer coordenadas de los puntos del borde
    y, x = np.where(edgeMatrix == 1)
    points = np.column_stack((x, y))

    # Aplicar PCA sobre los puntos de bordes
    pca = PCA(n_components=2)
    pca.fit(points)

    # El primer componente principal es la dirección del eje de máxima varianza
    first_component = pca.components_[0]
    angle_rad = np.arctan2(first_component[1], first_component[0])
    angle = angle_rad * 180 / math.pi

    # Get string orientation
    absangle = abs(angle)
    if 0 <= absangle <= 5:
        orientation = 'horizontal'
    elif 85 <= absangle <= 95:
        orientation = 'vertical'
    else:
        orientation = 'error'

    return angle, orientation

def exportData(X, Y1, Y2, col_names, path, name, format):
    # Check if Y2 is different from Y1
    if np.array_equal(Y1, Y2):
        Y2 = np.zeros(len(Y2))

    # Create a dataframe with the data
    data = pd.DataFrame({
        col_names[0]: X,
        col_names[1]: Y1,
        col_names[2]: Y2
    })

    # Just in case
    format = format.lower()

    # Verify format
    if format == 'excel':
        # Prepare the full path
        fullname = path + '/' + name + '.xlsx'
        filename = os.path.join(path, fullname)
        # Write the dataframe
        data.to_excel(filename, index=False)
        print(f'Data saved in Excel file: {filename}')

    elif format == 'csv':
        # Prepare the full path
        fullname = path + '/' + name + '.csv'
        filename = os.path.join(path, fullname)
        # Write the dataframe
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

def process_file(file_path, target_frequencies, valuesName, exportFormat):
    """Process a file to find the predicted MTF or NNPS values for the target frequencies using polynomial fitting."""
    try:
        # Read data from the file
        df = read_data(file_path, exportFormat)
        df = df.dropna(axis=1, how='all')

        # Fit polynomial and predict values
        predicted_values = fit_polynomial_and_predict(df, target_frequencies, 'Frequencies (1/mm)', valuesName)
        return dict(zip(target_frequencies, predicted_values))

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None