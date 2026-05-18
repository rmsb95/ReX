# ------------------------------------------------------ #
# NOISE POWER SPECTRUM & NORMALIZED NOISE POWER SPECTRUM #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome / Antonio Ortiz Lora
# Date: 05-2024
# Version: 2.0.0 - 2025/06
# Modified: Unified general + mammography NNPS calculation.
#           Modality is auto-detected from DICOM tag (0008,0060).
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


def _detect_modality(files):
    """
    Detect the imaging modality from the first valid DICOM file.

    Reads the DICOM tag Modality (0008,0060) to determine if the images
    correspond to mammography ('MG') or general radiography.

    Parameters
    ----------
    files : list of str
        List of DICOM file paths.

    Returns
    -------
    str
        'MG' for mammography, 'general' for any other modality.
    """
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            modality = getattr(ds, 'Modality', None)
            if modality:
                modality = modality.strip().upper()
                print(f"DICOM Modality detected: {modality}")
                if modality == 'MG':
                    return 'MG'
                else:
                    return 'general'
        except Exception as e:
            print(f"Warning: could not read modality from {f}: {e}")
    print("Warning: Modality tag not found in any file. Defaulting to 'general'.")
    return 'general'


def _calculate_mammography_offset(doseImage, cropSize, pixelSpacing):
    """
    Calculate the X offset so the ROI is centred at 60 mm from the
    left edge of the image, as required for mammography (IEC 62220-1-1:2015).

    Parameters
    ----------
    doseImage : np.ndarray
        The linearised dose image (2D).
    cropSize : int
        Size of the ROI (mm).
    pixelSpacing : float
        Pixel spacing (mm/px).

    Returns
    -------
    int
        Offset in pixels from the image centre (for cropImage).
    """
    roiSizePx = int(cropSize / pixelSpacing)
    imageWidth = doseImage.shape[1]
    desiredLeftEdge = int((60 - cropSize / 2) / pixelSpacing)
    targetCenterX = desiredLeftEdge + roiSizePx // 2
    imageCenterX = imageWidth // 2
    return targetCenterX - imageCenterX


def _get_nnps_modality_params(modality):
    """
    Return modality-specific default parameters for the NNPS calculation.

    Parameters
    ----------
    modality : str
        'MG' for mammography, 'general' otherwise.

    Returns
    -------
    dict with keys:
        cropSize : int
            Centered ROI size in mm.
        evaluateCentering : int
            1 = evaluate and auto-correct ROI centering, 0 = skip.
    """
    if modality == 'MG':
        return {
            'cropSize': 50,            # 50x50 mm for mammography
            'evaluateCentering': 0,    # Disabled: ROI is positioned manually at 60 mm
        }
    else:
        return {
            'cropSize': 125,           # 125x125 mm for general radiography
            'evaluateCentering': 1,    # Active: iterative centre-of-mass correction
        }


def calculateNNPS(path, conversion, a, b, exportFormat,
                  progress_callback=None, interaction_callback=None):
    """
    Calculate the Noise Power Spectrum (NPS) and Normalized NPS (NNPS).

    The imaging modality is automatically detected from the DICOM tag
    Modality (0008,0060). When mammography ('MG') is detected, the
    algorithm uses a smaller ROI (50 mm vs 125 mm), positions it at
    60 mm from the left edge, and disables iterative centering.

    Parameters
    ----------
    path : str
        Path to the directory containing DICOM files.
    conversion : str
        Type of conversion function ('linear' or 'log').
    a : float
        Coefficient 'a' of the response function.
    b : float
        Coefficient 'b' of the response function.
    exportFormat : str
        Export format ('excel' or 'csv').
    progress_callback : callable, optional
        Callback function to report progress (0-100).
    interaction_callback : callable, optional
        Callback function for user interaction when centering fails.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dataframe': pd.DataFrame with NNPS results
        - 'rois': dict mapping file paths to (x, y, height, width)
        - 'modality': str, detected modality ('MG' or 'general')
    """
    rois = {}

    # ---------------------
    # Section 1. Parameters
    # ---------------------
    files = ReX.find_dicom_files(path)

    # Auto-detect modality from DICOM files
    modality = _detect_modality(files)
    print(f"Operating in modality mode: {modality}")

    # Get modality-specific parameters
    params = _get_nnps_modality_params(modality)
    cropSize = params['cropSize']
    evaluateCentering = params['evaluateCentering']

    print(f"NNPS crop ROI: {cropSize} x {cropSize} mm")
    print(f"Centering evaluation: {'enabled' if evaluateCentering else 'disabled'}")

    # Small ROI size & step size (px) for NPS [IEC]
    roiSize = 256
    stepSize = 128

    # Counter of ROIs
    numROIs = 0

    # Offset (px) from center to trim ROI
    offsetCenterX = 0
    offsetCenterY = 0

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

        # --- Mammography-specific offset ---
        if modality == 'MG':
            offsetCenterX = _calculate_mammography_offset(doseImage, cropSize, pixelSpacing)
            offsetCenterY = 0
            print(f"Mammography offset: X = {offsetCenterX} px (60 mm from left edge)")

        # Crop centered ROI
        croppedImage, cropHeight, cropWidth, startX, startY = ReX.cropImage(
            doseImage, cropSize, cropSize,
            pixelSpacing, pixelSpacing,
            offsetCenterX, offsetCenterY
        )
        croppedImageArray = np.array(croppedImage)

        rois[file] = (startY, startX, cropHeight, cropWidth)
        print(f"ROI position - x: {startY} px ({startY * pixelSpacing:.1f} mm), "
              f"y: {startX} px ({startX * pixelSpacing:.1f} mm)")

        # Initialize dose array
        if i == 0:
            dose = np.zeros(len(files))

        # Get dose value to calculate NNPS later on
        dose[i] = croppedImageArray.mean()

        print(f"Dose of file {file} is {dose[i]} µGy")

        # Evaluate adequacy of the centering (not IEC)
        # Only for general radiography; disabled for mammography
        if evaluateCentering == 1:
            areThereLowerPixels = ReX.evaluateCentering(croppedImage, dose[i])

            if areThereLowerPixels:
                print(f'Warning: cropped ROI {i} has pixel values below threshold (80%). '
                      f'Try changing center OffSet.')

                maxIterations = 10
                iteration = 0
                while areThereLowerPixels:
                    if iteration >= maxIterations:
                        should_continue = True
                        if interaction_callback:
                            should_continue = interaction_callback(
                                f"No se ha encontrado el centro del ROI {i} "
                                f"tras {maxIterations} iteraciones."
                            )

                        if not should_continue:
                            raise Exception("Operación cancelada por el usuario.")

                        print(f'Warning: ROI {i} could not be centered after '
                              f'{maxIterations} iterations. The image may not be '
                              f'suitable for NPS analysis (e.g. MTF image). '
                              f'Proceeding with the current ROI.')
                        break

                    # Find the centre of mass of the image
                    centro = center_of_mass(doseImage)

                    offsetCenterY = int(centro[0] - (startY + cropHeight / 2))
                    offsetCenterX = int(centro[1] - (startX + cropWidth / 2))
                    print(doseImage.shape)
                    print(f"New offsetCenterX: {offsetCenterX}, New offsetCenterY: {offsetCenterY}")

                    # Recalculate
                    croppedImage, _, _, startX, startY = ReX.cropImage(
                        doseImage, cropSize, cropSize,
                        pixelSpacing, pixelSpacing,
                        offsetCenterX, offsetCenterY
                    )
                    rois[file] = (startY, startX, cropHeight, cropWidth)

                    # Evaluate again
                    areThereLowerPixels = ReX.evaluateCentering(croppedImage, dose[i])
                    iteration += 1

                # Recalculate after evaluation
                croppedImageArray = np.array(croppedImage)
                dose[i] = croppedImageArray.mean()

                if not areThereLowerPixels:
                    print(f'Great! Cropped ROI {i} is well centered now.')

            else:
                print(f'Cropped ROI {i} is well centered.')

        # Substracting the 2D polynomial of best fit
        residualImage = ReX.adjustedImage(croppedImage)

        # Create 256x256 px^2 ROIs. Overlapped 128 px.
        if numROIs == 0:
            nps_data = []

        for y in range(0, cropHeight - roiSize + 1, stepSize):
            for x in range(0, cropWidth - roiSize + 1, stepSize):
                # Get the ROI
                roi = residualImage[y:y + roiSize, x:x + roiSize]

                # Counting ROIs
                numROIs += 1

                # Calculate the square of the absolut value from the FT and stores it
                fft_result = fft.fft2(roi)
                nps_data.append(np.abs(fft_result) ** 2)

        i = i + 1

    for n in range(30, 50, 1):
        if progress_callback:
            progress_callback(n)
    if progress_callback:
        progress_callback(55)

    # ---------------------
    #   Section 3. 2D NPS
    # ---------------------
    # Summation of ROIs
    sum_nps_data = np.zeros_like(nps_data[0])

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
        np.arange(center - 7, center),       # Left
        np.arange(center + 1, center + 8)    # Right
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
            progress_callback(n)

    # Export NPS data
    ReX.exportData(verticalFrequencies, NPS_horizontal, NPS_vertical,
                   ['Frequencies (1/mm)', 'NPS Horizontal', 'NPS Vertical'],
                   path, 'NPS_data', exportFormat)
    # Export NNPS data
    ReX.exportData(verticalFrequencies, NNPS_horizontal, NNPS_vertical,
                   ['Frequencies (1/mm)', 'NNPS Horizontal', 'NNPS Vertical'],
                   path, 'NNPS_data', exportFormat)

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
    sample_step = 0.1  # Frequency step (1/mm)
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

    return {'dataframe': NNPS_to_DQE, 'rois': rois, 'modality': modality}