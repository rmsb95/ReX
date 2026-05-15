# ------------------------------------------------------ #
#           MODULATION TRANSFER FUNCTION (MTF)           #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 05-2024
# Version: 1.0.2 - 2025/06
# Modified: roiSizeA and roiSizeB are now parameters
#
# ---------------------
# Section 0. Imports
# ---------------------
import os

import matplotlib.pyplot as plt
import openpyxl
import glob
import pydicom
import math
import numpy as np
import pandas as pd
import src.ReXfunc as ReX


def calculateMTF(path, conversion, a, b, exportFormat, progress_callback=None, roiSizeA=50, roiSizeB=25):
    """
    Calculate the Modulation Transfer Function (MTF) for mammography.

    Parameters:
    -----------
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
        Callback function to report progress.
    roiSizeA : int, optional
        ROI height in mm for MTF (default: 50 mm per IEC 62220-1-1:2015 mammography).
    roiSizeB : int, optional
        ROI width in mm for MTF (default: 25 mm per IEC 62220-1-1:2015 mammography).

    Returns:
    --------
    dict
        Dictionary containing 'dataframe' with MTF results and 'rois' with ROI positions.
    """
    rois = {}

    # ---------------------
    # Section 1. Parameters
    # ---------------------
    files = ReX.find_dicom_files(path)

    # Flags to determine if both orientations have been obtained
    verticalFlag = 0
    horizontalFlag = 0

    # ROI size (mm) for non-uniformity correction (IEC 62220-1-1:2015 mammography)
    # 100 mm x 100 mm para el inicio del cálculo
    roiSizeAUC = 100
    roiSizeBUC = 100

    # Offset (px) from center to trim ROI
    # Para mamografía: 60 mm del borde izquierdo, centrado en dirección mayor
    offsetCenterX = 0
    offsetCenterY = 0

    # Non-uniformity correction (IEC 62220-1-1:2015)
    isNeeded = 0

    # Edge detection algorithm
    algorithmEdge = 'Roberts'

    # ---------------------
    # Section 2. Image Loop
    # ---------------------
    for n in range(1, 20, 1):
        progress_callback(n)

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
            print("Se ha aplicado FR lineal")

        elif conversion == 'log':
            doseImage = np.exp((dicomImage.astype(float) - b) / a)
            print("Se ha aplicado FR log")

        # Substitute negative values -> zeros
        doseImage[doseImage < 0] = 0

        # Calcular offset para posicionar ROI a 60 mm del borde izquierdo
        roiSizeUCPx = int(roiSizeAUC / pixelSpacing)
        imageHeight = doseImage.shape[0]
        imageWidth = doseImage.shape[1]

        # Posición deseada: 60 mm del borde izquierdo
        desiredLeftEdge = int((60-roiSizeAUC/2) / pixelSpacing)
        targetCenterX = desiredLeftEdge + roiSizeUCPx // 2
        imageCenterX = imageWidth // 2
        offsetCenterX = targetCenterX - imageCenterX

        # Centrado verticalmente (en la dirección mayor)
        offsetCenterY = 0

        # -------------------------------------
        # Section 2*. Non-uniformity correction
        # -------------------------------------
        # Crop the ROI for the correction (100x100 mm)
        croppedImage, _, _, crop_start_row, crop_start_col = ReX.cropImage(doseImage, roiSizeBUC, roiSizeAUC,
                                                                           pixelSpacing, pixelSpacing,
                                                                           offsetCenterX, offsetCenterY)

        # Luego, en lugar de plt.show():
        plt.figure(figsize=(8, 6))
        plt.imshow(croppedImage, cmap='gray')
        plt.title('ROI de corrección de no uniformidad (100x100 mm)')
        plt.colorbar(label='Dosis')
        plt.tight_layout()

        # Guardar imagen en lugar de mostrar
        output_path = os.path.join(path, f'ROI_correction_{i}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f'Imagen guardada en: {output_path}')
        if isNeeded == 1:
            # Apply non-uniformity correction
            corrImage = ReX.correctedImage(croppedImage)

        elif isNeeded == 0:
            # Not needed
            corrImage = croppedImage

        # -------------------
        # Section 3. MTF ROI
        # -------------------
        # Crop 50x25 mm MTF ROI (roiSizeA x roiSizeB)
        croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(corrImage, roiSizeB, roiSizeA, pixelSpacing,
                                                                       pixelSpacing, 0, 0)
        roi_height, roi_width = croppedROI.shape

        # -------------------------
        # Section 4. Edge Detection
        # -------------------------
        angle, orientation = ReX.edgeDetection(croppedROI)

        print(f'Angle: {angle}')
        print(f'Orientation: {orientation}')

        absAngle = abs(angle)

        # Adjust image to horizontal orientation for next steps
        for n in range(21, 40, 1):
            progress_callback(n)

        if orientation == 'vertical':

            verticalFlag = 1

            croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(corrImage, roiSizeA, roiSizeB, pixelSpacing,
                                                                           pixelSpacing, 0,
                                                                           0)
            roi_height, roi_width = croppedROI.shape

            # Angle is recalculated after cropping
            angle_after_new_cropping, _ = ReX.edgeDetection(croppedROI)
            NAngle = abs(90 - abs(angle_after_new_cropping))

            if absAngle > 90:
                croppedROI = np.rot90(croppedROI)
            elif absAngle < 90:
                croppedROI = np.flipud(np.rot90(croppedROI))

        elif orientation == 'horizontal':

            horizontalFlag = 1

            NAngle = absAngle

        if angle > 0:
            croppedROI = np.fliplr(croppedROI)

        x = crop_start_col + roi_start_col
        y = crop_start_row + roi_start_row
        rois[file] = (x, y, roi_width, roi_height)
        print(f"orientation: {orientation}, x: {x}, y: {y}, width: {roi_width}, height: {roi_height}")

        # -------------------------------------
        # Section 5. Edge Spread Function (ESF)
        # -------------------------------------
        N = round(1 / math.tan(math.radians(NAngle)))
        totalColumns = roiSizeB / pixelSpacing
        numGroups = math.floor(totalColumns / N)

        # Initialize to store each group ESF
        all_ESFs = np.zeros((croppedROI.shape[0] * N, numGroups))

        for n in range(41, 60, 1):
            progress_callback(n)

        # Calculation of ESF for each group of N columns
        for k in range(1, numGroups + 1):
            startCol = (k - 1) * N
            endCol = startCol + N

            edgeRegion = croppedROI[:, startCol:endCol]

            current_ESF = np.zeros((edgeRegion.shape[0] * N))

            for col in range(N):
                current_ESF[col::N] = edgeRegion[:, col]

            displacement = (k - 1) * N

            if displacement != 0:
                despl_ESF = np.zeros_like(current_ESF)
                despl_ESF[displacement:] = current_ESF[:-displacement]
                current_ESF = despl_ESF

            all_ESFs[:, k - 1] = current_ESF

        # Calculation of average ESF
        average_ESF = np.mean(all_ESFs, axis=1)

        # Slice the ESF to avoid unuseful values due to displacement (not IEC)
        start = 499
        average_ESF = average_ESF[start:]

        for n in range(61, 70, 1):
            progress_callback(n)

        # -------------------------------------
        # Section 6. Line Spread Function (LSF)
        # -------------------------------------
        kernel = np.array([-1, 0, 1])
        LSF = np.convolve(average_ESF, kernel, mode='valid')

        # ---------------------------------------------
        # Section 7. Modulation Transfer Function (MTF)
        # ---------------------------------------------
        fNyq = 1 / (2 * pixelSpacing)

        MTF = np.abs(np.fft.fft(LSF))
        MTF = MTF[:len(MTF) // 2]
        MTF = MTF / MTF[0]

        LSFlength = len(LSF)
        frequencySpacing = 1 / (LSFlength * pixelSpacing * N)

        frequencies = np.linspace(-0.5 * N / pixelSpacing, 0.5 * N / pixelSpacing - frequencySpacing, LSFlength)

        frequencies = np.fft.fftshift(frequencies)
        frequencies = frequencies[:len(frequencies) // 2]

        # ------------------------------
        # Section 7.b. Smoothing the MTF
        # ------------------------------
        for n in range(71, 85, 1):
            progress_callback(n)

        fint = 0.01 / pixelSpacing

        MTF_smoothed = np.zeros(len(frequencies))

        for i, f in enumerate(frequencies):
            lowerLimit = f - 0.5 * fint
            upperLimit = f + 0.5 * fint

            mask = (frequencies >= lowerLimit) & (frequencies <= upperLimit)
            MTF_smoothed[i] = np.mean(MTF[mask])

        MTF_smoothed = MTF_smoothed / MTF_smoothed[0]

        # --------------------------
        # Section 8. Export MTF data
        # --------------------------
        for n in range(86, 95, 1):
            progress_callback(n)

        validIndexes = frequencies <= fNyq
        frequencies = frequencies[validIndexes]
        MTF_smoothed = MTF_smoothed[validIndexes]

        name = 'MTF_' + orientation
        ReX.exportData(frequencies, MTF_smoothed, MTF_smoothed, ['Frequencies (1/mm)', 'MTF', ''], path, name,
                       exportFormat)

        i = i + 1

    # --------------------------
    # Section 9. Combine MTFs
    # --------------------------
    print("Defining Target frequencies.")
    sample_step = 0.1
    target_frequencies = np.arange(sample_step, fNyq + sample_step, sample_step)
    target_frequencies = target_frequencies[target_frequencies <= fNyq]
    print("Target frequencies defined.")

    results = {freq: {} for freq in target_frequencies}
    print("Result dictionary initialized")

    print("Bucle for")
    for file_name in os.listdir(path):
        print(file_name)
        file_path = os.path.join(path, file_name)
        if "MTF_vertical" in file_name:
            print("MTF vertical")
            data = ReX.process_file(file_path, target_frequencies, 'MTF', exportFormat)
            for freq, value in data.items():
                results[freq]['MTF Vertical'] = value
        elif "MTF_horizontal" in file_name:
            print("MTF horizontal")
            data = ReX.process_file(file_path, target_frequencies, 'MTF', exportFormat)
            for freq, value in data.items():
                results[freq]['MTF Horizontal'] = value

    final_df = pd.DataFrame(results).T
    final_df.index.name = 'Frequencies (1/mm)'

    column_order = ['MTF Horizontal', 'MTF Vertical']
    final_df = final_df[[col for col in column_order if col in final_df.columns]]

    if verticalFlag == 1 and horizontalFlag == 1:
        if exportFormat == 'excel':
            output_file_path = os.path.join(path, 'MTF_to_DQE.xlsx')
            final_df.to_excel(output_file_path)
        elif exportFormat == 'csv':
            output_file_path = os.path.join(path, 'MTF_to_DQE.csv')
            final_df.to_excel(output_file_path)

        print(f'Data saved as: {output_file_path}')

    progress_callback(100)
    progress_callback(0)
    return {'dataframe': final_df, 'rois': rois}
