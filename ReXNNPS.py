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
import glob
import pydicom
import math
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import ReXfunc as ReX


def calculateNNPS(path, conversion, a, b, exportFormat, progress_callback, log_queue):
    # ---------------------
    # Section 1. Parameters
    # ---------------------
    # path = 'C:/Users/rafa_/OneDrive/Residencia/01. R1/02. Radiodiagnóstico/Aplicaciones/Cálculo NPS/NNPS/Alta dosis/'
    files = glob.glob(os.path.join(path, '*.DCM'))
    progress_callback(0)
    log_queue.put("NPS calculation started.")
    # Export format (saved in path): 'excel' or 'csv'
    # exportFormat = 'excel'

    # Centered ROI size (mm)
    cropSize = 125

    # Small ROI size & step size (px) for NPS [IEC]
    roiSize = 256
    stepSize = 128

    # Counter of ROIs
    numROIs = 0

    # Option to assess the adequacy of the ROI centering (not IEC)
    evaluateCentering = 0

    # Offset (px) from center to trim ROI
    offsetCenterX = 0
    offsetCenterY = 0

    # ---------------------
    # Section 2. Image Loop
    # ---------------------
    progress_callback(20)
    i = 0
    for file in files:

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

        # Linearize data (from Mean Pixel Value to Dose)
        if conversion == 'linear':
            doseImage = (dicomImage.astype(float) - b) / a

        elif conversion == 'log':
            doseImage = math.exp((dicomImage.astype(float) - b) / a)

        # Crop 125x125 mm^2 centered ROI
        croppedImage, cropHeight, cropWidth = ReX.cropImage(doseImage, cropSize, cropSize, pixelSpacing, pixelSpacing,
                                                            offsetCenterX, offsetCenterY)
        croppedImageArray = np.array(croppedImage)

        # Initialize dose array
        if i == 0:
            dose = np.zeros(len(files))

        # Get dose value to calculate NNPS later on
        dose[i] = croppedImageArray.mean()

        # Evaluate adequacy of the centering (not IEC)
        if evaluateCentering == 1:
            areThereLowerPixels = ReX.evaluateCentering(croppedImage, dose[i])

            if areThereLowerPixels:
                print(f'Warning: cropped ROI {i} has pixel values below threshold (80%). Try changing center OffSet.')

                while areThereLowerPixels:
                    offsetCenterX = int(input("Enter a new value (in pixels) for offsetCenterX: "))
                    offsetCenterY = int(input("Enter a new value (in pixels) for offsetCenterY: "))

                    # Recalculate
                    croppedImage, _, _ = ReX.cropImage(doseImage, cropSize, cropSize, pixelSpacing, pixelSpacing,
                                                       offsetCenterX,
                                                       offsetCenterY)

                    # Evaluate again
                    areThereLowerPixels = ReX.evaluateCentering(croppedImage, dose[i])

                # Recalculate after evaluation
                croppedImageArray = np.array(croppedImage)
                dose[i] = croppedImageArray.mean()

                print(f'Great! Cropped ROI {i} is well centered now.')

                # TO BE IMPROVED: AUTOMATE CENTERING. GET THE MAX INDEX WITH 1 IN LOWERPIXEL MATRIX AND ADJUST OFFSET
            else:
                print(f'Cropped ROI {i} is well centered.')

        # Substracting the 2D polynomial of best fit
        residualImage = ReX.adjustedImage(croppedImage)
        # TO INVESTIGATE: IS IT POSSIBLE TO GET BETTER FITTING?

        progress_callback(40)

        # Create 256x256 px^2 ROIs. Overlapped 128 px.
        if numROIs == 0:
            nps_data = []

        log_queue.put("Creating ROIs.")
        for y in range(0, cropHeight - roiSize + 1, stepSize):
            for x in range(0, cropWidth - roiSize + 1, stepSize):
                # Get the ROI
                roi = residualImage[y:y + roiSize, x:x + roiSize]

                # Counting ROIs
                numROIs += 1

                # Calculate the square of the absolut value from the FT and stores it
                fft_result = fft.fft2(roi)
                nps_data.append(np.abs(fft_result) ** 2)

        i = i + 1  # Para el final del bucle for

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
    log_queue.put(f"Mean estimated dose: { meanDose }.")

    # 2D NPS
    log_queue.put("Calculating 2D NPS.")
    NPS = pixelSpacing * pixelSpacing / (numROIs * 256 * 256) * sum_nps_data
    NPS = np.fft.fftshift(NPS[0:256, 0:256])

    # ---------------------
    #   Section 4. 1D NPS
    # ---------------------
    log_queue.put("Calculating 1D NPS.")
    progress_callback(60)
    # Defining some parameters
    fint = 0.01 / pixelSpacing # Binning frequency (IEC)
    NPS_dim = NPS.shape[0]
    center = NPS_dim / 2

    # Calculation of spacial frequencies in the sense of distances
    frequenciesGrid = np.linspace(-center, center-1, NPS_dim) / (NPS_dim * pixelSpacing)
    X, Y = np.meshgrid(frequenciesGrid, frequenciesGrid)
    frequenciesRadial = np.sqrt(X**2 + Y**2)

    # Select 14 vertical lines & 14 horizontal lines, but not the center
    lineIndexes = np.concatenate([
        np.arange(center - 7, center),  # Left
        np.arange(center + 1, center + 8)   # Right
    ]).astype(int)

    # Initialize the vector to store the averages of NPS based on spatial frequencies
    frequencies = np.linspace(0, np.max(frequenciesRadial), int(center))
    NPS_vertical = np.zeros_like(frequencies)
    NPS_horizontal = np.zeros_like(frequencies)

    progress_callback(80)
    log_queue.put("Smoothing 1D NPS.")
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
    log_queue.put("Calculating NNPS.")
    progress_callback(90)
    NNPS_vertical = NPS_vertical / (meanDose ** 2)
    NNPS_horizontal = NPS_horizontal / (meanDose ** 2)

    # ---------------------
    #   Section 6. Export
    # ---------------------
    # Export NPS data
    ReX.exportData(verticalFrequencies, NPS_vertical, NPS_horizontal, ['Frequencies','NPS Vertical', 'NPS Horizontal'], path, 'NPS_data', exportFormat)
    # Export NNPS data
    ReX.exportData(verticalFrequencies, NNPS_vertical, NNPS_horizontal, ['Frequencies','NNPS Vertical', 'NNPS Horizontal'], path, 'NNPS_data', exportFormat)

    progress_callback(100)
    log_queue.put(f"Finished! Data files saved in { path }")
    progress_callback(0)