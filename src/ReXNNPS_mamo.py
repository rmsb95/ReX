# ------------------------------------------------------ #
# NOISE POWER SPECTRUM & NORMALIZED NOISE POWER SPECTRUM #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 05-2024
# Version: 1.0
#
# ---------------------
# Section 0. Imports
# ---------------------
import os
import openpyxl
import glob
import pydicom
import math
import pandas as pd
import numpy as np
import numpy.fft as fft
from scipy.ndimage import center_of_mass
import src.ReXfunc as ReX


def calculateNNPS(path, conversion, a, b, exportFormat, progress_callback=None, interaction_callback=None):
    rois = {}
    # ---------------------
    # Section 1. Parameters
    # ---------------------
    files = ReX.find_dicom_files(path)

    # Centered ROI size (mm) - ADAPTADO PARA MAMOGRAFÍA: 50x50 mm
    cropSize = 50

    # Small ROI size & step size (px) for NPS [IEC]
    roiSize = 256
    stepSize = 128

    # Counter of ROIs
    numROIs = 0

    # Option to assess the adequacy of the ROI centering (not IEC)
    evaluateCentering = 0  # Desactivado para posicionamiento manual

    # Offset (px) from center to trim ROI
    # Para mamografía: 60 mm del borde izquierdo, centrado en dirección mayor
    offsetCenterX = 0  # Se calculará basado en la posición deseada
    offsetCenterY = 0  # Centrado verticalmente

    # ---------------------
    # Section 2. Image Loop
    # ---------------------
    for n in range(1, 20, 1):
        if progress_callback:
            progress_callback(n)

    i = 0
    for file in files:
        print(f"Working with {file}")
        # Read dcm file
        dicomFile = pydicom.dcmread(file)
        dicomImage = dicomFile.pixel_array

        # Get dicom properties
        if 'PixelSpacing' in dicomFile:
            pixelSpacing = float(dicomFile.PixelSpacing[1])
        elif 'ImagerPixelSpacing' in dicomFile:
            pixelSpacing = float(dicomFile.ImagerPixelSpacing[1])
        else:
            print("PixelSpacing is not available in DICOM file. Default value (0.1 mm) will be used.")
            pixelSpacing = 0.1

        print(f"Pixel size is {pixelSpacing}")

        # Linearize data (from Mean Pixel Value to Dose)
        if conversion == 'linear':
            doseImage = (dicomImage.astype(float) - b) / a
        elif conversion == 'log':
            doseImage = np.exp((dicomImage.astype(float) - b) / a)

        # Calcular offset para posicionar ROI a 60 mm del borde izquierdo
        # y centrada en la dirección mayor (vertical)
        roiSizePx = int(cropSize / pixelSpacing)  # Tamaño ROI en píxeles
        imageHeight = doseImage.shape[0]
        imageWidth = doseImage.shape[1]

        # Posición deseada: centro de la roi a 60 mm del borde izquierdo
        desiredLeftEdge = int((60-(cropSize/2)) / pixelSpacing)
        targetCenterX = desiredLeftEdge + roiSizePx // 2
        imageCenterX = imageWidth // 2
        offsetCenterX = targetCenterX - imageCenterX

        # Centrado verticalmente (en la dirección mayor)
        offsetCenterY = 0

        # Crop 50x50 mm ROI con posicionamiento específico
        croppedImage, cropHeight, cropWidth, startX, startY = ReX.cropImage(
            doseImage, cropSize, cropSize, pixelSpacing, pixelSpacing,
            offsetCenterX, offsetCenterY)
        croppedImageArray = np.array(croppedImage)

        rois[file] = (startY, startX, cropHeight, cropWidth)
        print(
            f"ROI position - x: {startY} px ({startY * pixelSpacing:.1f} mm), y: {startX} px ({startX * pixelSpacing:.1f} mm)")

        # Initialize dose array
        if i == 0:
            dose = np.zeros(len(files))

        # Get dose value
        dose[i] = croppedImageArray.mean()
        print(f"Dose of file {file} is {dose[i]} µGy")

        # Substracting the 2D polynomial of best fit
        residualImage = ReX.adjustedImage(croppedImage)

        # Create 256x256 px ROIs with 128 px overlap
        if numROIs == 0:
            nps_data = []

        for y in range(0, cropHeight - roiSize + 1, stepSize):
            for x in range(0, cropWidth - roiSize + 1, stepSize):
                roi = residualImage[y:y + roiSize, x:x + roiSize]
                numROIs += 1
                fft_result = fft.fft2(roi)
                nps_data.append(np.abs(fft_result) ** 2)

        i = i + 1

    # Resto del código permanece igual...

    for n in range(30, 50, 1):
        if progress_callback:
            progress_callback(n)
    if progress_callback:
        progress_callback(55)
    # ---------------------
    #   Section 3. 2D NPS
    # ---------------------
    # Summation of ROIs
    # Initialize an accumulated sum matrix with the same dimension as the elements in NPS
    sum_nps_data = np.zeros_like(nps_data[0])

    # Iterate through each element of npsData and add its content to the total sum
    for roi_data in nps_data:
        sum_nps_data += roi_data

    # Average dose of the images
    meanDose = np.mean(dose[:])
    print(f'The average dose of the images is: {meanDose:.2f} µGy.')

    if progress_callback:
        progress_callback(60)

    # 2D NPS
    NPS = pixelSpacing * pixelSpacing / (numROIs * 256 * 256) * sum_nps_data
    NPS = np.fft.fftshift(NPS[0:256, 0:256])

    # ---------------------
    #   Section 4. 1D NPS
    # ---------------------
    if progress_callback:
        progress_callback(65)

    # Defining some parameters
    fint = 0.01 / pixelSpacing  # Binning frequency (IEC)
    NPS_dim = NPS.shape[0]
    center = NPS_dim / 2

    # Calculation of spacial frequencies in the sense of distances
    frequenciesGrid = np.linspace(-center, center - 1, NPS_dim) / (NPS_dim * pixelSpacing)
    X, Y = np.meshgrid(frequenciesGrid, frequenciesGrid)
    frequenciesRadial = np.sqrt(X ** 2 + Y ** 2)

    if progress_callback:
        progress_callback(70)

    # Select 14 vertical lines & 14 horizontal lines, but not the center
    lineIndexes = np.concatenate([
        np.arange(center - 7, center),  # Left
        np.arange(center + 1, center + 8)  # Right
    ]).astype(int)

    if progress_callback:
        progress_callback(75)

    # Initialize the vector to store the averages of NPS based on spatial frequencies
    frequencies = np.linspace(0, np.max(frequenciesRadial), int(center))
    NPS_vertical = np.zeros_like(frequencies)
    NPS_horizontal = np.zeros_like(frequencies)

    if progress_callback:
        progress_callback(80)
    for i, f in enumerate(frequencies):
        lowerLimit = f - 0.5 * fint
        upperLimit = f + 0.5 * fint

        # Select positions that fall within the frequency interval
        # Creates a logical matrix of the same size as NPS where the condition is met
        mask = (frequenciesRadial >= lowerLimit) & (frequenciesRadial <= upperLimit)

        # Average the values of the 14 lines for the selected spatial frequencies
        verticalMask = mask & np.isin(X, frequenciesGrid[lineIndexes])
        horizontalMask = mask & np.isin(Y, frequenciesGrid[lineIndexes])
        NPS_vertical[i] = np.mean(NPS[verticalMask])
        NPS_horizontal[i] = np.mean(NPS[horizontalMask])
        print(f"Finished mean number {i}!")

    if progress_callback:
        progress_callback(85)

    # Removing NaN values
    validValues = ~np.isnan(NPS_vertical)
    verticalFrequencies = frequencies[validValues]
    NPS_vertical = NPS_vertical[validValues]

    validValues = ~np.isnan(NPS_horizontal)
    horizontalFrequencies = frequencies[validValues]
    NPS_horizontal = NPS_horizontal[validValues]

    # ---------------------
    #   Section 5. NNPS
    # ---------------------
    if progress_callback:
        progress_callback(90)
    NNPS_vertical = NPS_vertical / (meanDose ** 2)
    NNPS_horizontal = NPS_horizontal / (meanDose ** 2)

    # ---------------------
    #   Section 6. Export
    # ---------------------
    for n in range(90, 99, 1):
        if progress_callback:
            progress_callback(i)
    # Export NPS data
    ReX.exportData(verticalFrequencies, NPS_horizontal, NPS_vertical,
                   ['Frequencies (1/mm)', 'NPS Horizontal', 'NPS Vertical'], path, 'NPS_data', exportFormat)
    # Export NNPS data
    ReX.exportData(verticalFrequencies, NNPS_horizontal, NNPS_vertical,
                   ['Frequencies (1/mm)', 'NNPS Horizontal', 'NNPS Vertical'], path, 'NNPS_data', exportFormat)

    if progress_callback:
        progress_callback(100)
        progress_callback(0)

    # ------------------------
    #   Section 7. Data to DQE
    # ------------------------
    # It prepares data for DQE calculation
    # Calculate Nyquist frequency
    fNyq = 1 / (2 * pixelSpacing)

    # Create target frequencies array
    sample_step = 0.1  # Frequency step (1/mm) añadido por AOL
    target_frequencies = np.arange(sample_step, fNyq + sample_step, sample_step)
    target_frequencies = target_frequencies[target_frequencies <= fNyq]

    # Read data
    if exportFormat == 'excel':
        file_path = os.path.join(path, 'NNPS_data.xlsx')
        df = pd.read_excel(file_path)
    elif exportFormat == 'csv':
        file_path = os.path.join(path, 'NNPS_data.csv')
        df = pd.read_csv(file_path)

    # Get closest NNPS values for each target frequency
    # results = []
    # for freq in target_frequencies:
    #     closest_row = ReX.find_closest(df, freq, 'Frequencies (1/mm)')
    #     results.append({
    #         'Frequencies (1/mm)': freq,
    #         'NNPS Horizontal': closest_row['NNPS Horizontal'],
    #         'NNPS Vertical': closest_row['NNPS Vertical']
    #     })

    # Initialize results dictionary
    results = {freq: {} for freq in target_frequencies}
    print("Result dictionary initialized")

    NNPS_hor_df = ReX.process_file(file_path, target_frequencies, 'NNPS Horizontal', exportFormat)
    for freq, value in NNPS_hor_df.items():
        results[freq]['NNPS Horizontal'] = value

    NNPS_ver_df = ReX.process_file(file_path, target_frequencies, 'NNPS Vertical', exportFormat)
    for freq, value in NNPS_ver_df.items():
        results[freq]['NNPS Vertical'] = value

    # Create the dataframe
    NNPS_to_DQE = pd.DataFrame(results).T
    NNPS_to_DQE.index.name = 'Frequencies (1/mm)'

    # Save final DataFrame
    if exportFormat == 'excel':
        output_file_path = os.path.join(path, 'NNPS_to_DQE.xlsx')
        NNPS_to_DQE.to_excel(output_file_path)
    elif exportFormat == 'csv':
        output_file_path = os.path.join(path, 'NNPS_to_DQE.csv')
        NNPS_to_DQE.to_csv(output_file_path)

    return {'dataframe': NNPS_to_DQE, 'rois': rois}
