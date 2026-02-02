# ------------------------------------------------------ #
#           MODULATION TRANSFER FUNCTION (MTF)           #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 05-2024
# Version: 1.0.1 - 2025/05/16
#
# ---------------------
# Section 0. Imports
# ---------------------
import os
import openpyxl
import glob
import pydicom
import math
import numpy as np
import pandas as pd
import src.ReXfunc as ReX

def calculateMTF(path, conversion, a, b, exportFormat, progress_callback=None):
    rois = {}

    # ---------------------
    # Section 1. Parameters
    # ---------------------
    files = ReX.find_dicom_files(path)

    # Flags to determine if both orientations have been obtained
    verticalFlag = 0
    horizontalFlag = 0

    # ROI size (mm) (IEC 62220-1-1:2015) considering vertical position of the edge
    roiSizeA = 100  # width
    roiSizeB = 50   # height

    # ROI size (mm) for non-uniformity correction (IEC 62220-1-1:2015)
    # It should be at least 1.5 times bigger than the other one
    roiSizeAUC = 150
    roiSizeBUC = 150

    # Offset (px) from center to trim ROI
    offsetCenterX = 0
    offsetCenterY = 0

    # Non-uniformity correction (IEC 62220-1-1:2015)
    # TBR: CHECK ALGORITHM
    isNeeded = 0

    # Edge detection algorithm
    # TBR: it's not possible to change it right now.
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

        # -------------------------------------
        # Section 2*. Non-uniformity correction
        # -------------------------------------
        # Crop the ROI for the correction
        croppedImage, _, _, crop_start_row, crop_start_col = ReX.cropImage(doseImage, roiSizeBUC, roiSizeAUC, pixelSpacing, pixelSpacing,
                                                            offsetCenterX, offsetCenterY)

        if isNeeded == 1:
            # Apply non-uniformity correction
            # Calculation of 2D best fit (S)
            # Corrected image = Original Image / S * S_avg
            corrImage = ReX.correctedImage(croppedImage)

        elif isNeeded == 0:
            # Not needed
            corrImage = croppedImage


        # -------------------
        # Section 3. MTF ROI
        # -------------------
        # Crop 100 mm x 50 mm central ROI for MTF
        croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(corrImage, roiSizeB, roiSizeA, pixelSpacing, pixelSpacing, offsetCenterX, offsetCenterY)
        roi_height, roi_width = croppedROI.shape

        # -------------------------
        # Section 4. Edge Detection
        # -------------------------
        # Function to calculate angle and orientation
        angle, orientation = ReX.edgeDetection(croppedROI)

        print(f'Angle: {angle}')
        print(f'Orientation: {orientation}')

        absAngle = abs(angle)

        # Adjust image to horizontal orientation for next steps
        for n in range(21, 40, 1):
            progress_callback(n)

        if orientation == 'vertical':

            verticalFlag = 1

            croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(corrImage, roiSizeA, roiSizeB, pixelSpacing, pixelSpacing, offsetCenterX,
                                             offsetCenterY)
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
        # Calculation of N columns for the ESF
        N = round(1 / math.tan(math.radians(NAngle)))
        # Estimate number of groups made of N columns
        totalColumns = roiSizeB / pixelSpacing
        numGroups = math.floor(totalColumns / N)

        # Initialize to store each group ESF
        all_ESFs = np.zeros((croppedROI.shape[0] * N, numGroups))

        for n in range(41, 60, 1):
            progress_callback(n)

        # Calculation of ESF for each group of N columns
        for k in range(1, numGroups + 1):
            # Calculate first index for each group of N columns
            startCol = (k - 1) * N  # Esto queda modificado respecto a MATLAB
            endCol = startCol + N   # Esto queda modificado respecto a MATLAB

            # Get region for current group
            edgeRegion = croppedROI[:, startCol:endCol]

            # Initialize current ESF vector
            current_ESF = np.zeros((edgeRegion.shape[0] * N))

            # Calculate current ESF vector
            for col in range(N):
                current_ESF[col::N] = edgeRegion[:, col]

            # Vector displacement for edge averaging
            displacement = (k - 1) * N

            # It does the displacement only when different from 0 (avoids error)
            if displacement != 0:
                despl_ESF = np.zeros_like(current_ESF)
                despl_ESF[displacement:] = current_ESF[:-displacement]
                current_ESF = despl_ESF

            # It stores current ESF in the matrix
            all_ESFs[:, k - 1] = current_ESF

        # Calculation of average ESF
        average_ESF = np.mean(all_ESFs, axis=1)

        # Slice the ESF to avoid unuseful values due to displacement (not IEC)
        start = 499     # Max(N * k) = 500 -> 500 - 1 (python index) = 499
        average_ESF = average_ESF[start:]

        for n in range(61, 70, 1):
            progress_callback(n)

        # -------------------------------------
        # Section 6. Line Spread Function (LSF)
        # -------------------------------------
        # Define kernel for convolution
        kernel = np.array([-1, 0, 1])

        # Convolution
        LSF = np.convolve(average_ESF, kernel, mode='valid')


        # ---------------------------------------------
        # Section 7. Modulation Transfer Function (MTF)
        # ---------------------------------------------
        # Calculate Nyquist frequency
        fNyq = 1/(2*pixelSpacing)

        # Calculate the FFT from LSF
        MTF = np.abs(np.fft.fft(LSF))

        # Take the positive half
        MTF = MTF[:len(MTF)//2]

        # Normalize MTF
        MTF = MTF / MTF[0]

        # Calculate LSF length to create frequency vector
        LSFlength = len(LSF)

        # Frequency "step" between positions in the array
        frequencySpacing = 1 / (LSFlength * pixelSpacing * N)

        # Create the frequency vector
        frequencies = np.linspace(-0.5 * N / pixelSpacing, 0.5 * N / pixelSpacing - frequencySpacing, LSFlength)

        # Take the positive half
        frequencies = np.fft.fftshift(frequencies)
        frequencies = frequencies[:len(frequencies)//2]

        # ------------------------------
        # Section 7.b. Smoothing the MTF
        # ------------------------------
        for n in range(71, 85, 1):
            progress_callback(n)

        fint = 0.01 / pixelSpacing  # Binning frequency (IEC)

        # Initialize the new MTF matrix
        MTF_smoothed = np.zeros(len(frequencies))

        for i, f in enumerate(frequencies):
            lowerLimit = f - 0.5 * fint
            upperLimit = f + 0.5 * fint

            # Select positions that fall within the frequency interval
            # Creates a logical matrix of the same size as NPS where the condition is met
            mask = (frequencies >= lowerLimit) & (frequencies <= upperLimit)

            # Average the values of the 14 lines for the selected spatial frequencies
            MTF_smoothed[i] = np.mean(MTF[mask])

        # Normalize the smoothed MTF
        MTF_smoothed = MTF_smoothed / MTF_smoothed[0]

        # --------------------------
        # Section 8. Export MTF data
        # --------------------------
        for n in range(86, 95, 1):
            progress_callback(n)

        # Slice frequencies below Nyquist frequency
        validIndexes = frequencies <= fNyq
        frequencies = frequencies[validIndexes]
        MTF_smoothed = MTF_smoothed[validIndexes]

        # Actual export
        name = 'MTF_' + orientation
        ReX.exportData(frequencies, MTF_smoothed, MTF_smoothed, ['Frequencies (1/mm)', 'MTF', ''], path, name, exportFormat)

        i = i + 1

    # --------------------------
    # Section 9. Combine MTFs
    # --------------------------
    # If MTF has been calculated for vertical and horizontal orientation
    # it creates a file with the values prepared to calculate DQE later

    # Define target frequencies
    print("Defining Target frequencies.")
    sample_step = 0.1               # Frequency step (1/mm) añadido por AOL
    target_frequencies = np.arange(sample_step, fNyq + sample_step, sample_step)
    target_frequencies = target_frequencies[target_frequencies <= fNyq]
    print("Target frequencies defined.")

    # Initialize results dictionary
    results = {freq: {} for freq in target_frequencies}
    print("Result dictionary initialized")

    # Process each file in the directory
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

    # Create DataFrame from results
    final_df = pd.DataFrame(results).T
    final_df.index.name = 'Frequencies (1/mm)'

    if verticalFlag == 1 and horizontalFlag == 1:
        # Save final DataFrame
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
