# ------------------------------------------------------ #
#           MODULATION TRANSFER FUNCTION (MTF)           #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome / Antonio Ortiz Lora
# Date: 05-2024
# Version: 2.0.0 - 2025/06
# Modified: Unified general + mammography MTF calculation.
#           Modality is auto-detected from DICOM tag (0008,0060).
#           roiSizeA and roiSizeB are now parameters with
#           modality-specific defaults.
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


def _get_modality_params(modality, roiSizeA=None, roiSizeB=None):
    """
    Return modality-specific default parameters for the MTF calculation.

    If roiSizeA or roiSizeB are explicitly provided (not None), they
    override the modality defaults.

    Parameters
    ----------
    modality : str
        'MG' for mammography, 'general' otherwise.
    roiSizeA : int or None
        User-provided ROI dimension A (mm). None = use default.
    roiSizeB : int or None
        User-provided ROI dimension B (mm). None = use default.

    Returns
    -------
    dict with keys:
        roiSizeA, roiSizeB, roiSizeAUC, roiSizeBUC
    """
    if modality == 'MG':
        # IEC 62220-1-2:2015 mammography defaults
        default_A = 50   # height (mm)
        default_B = 25   # width (mm)
        _roiA = roiSizeA if roiSizeA is not None else default_A
        _roiB = roiSizeB if roiSizeB is not None else default_B
        # Non-uniformity correction ROI: fixed 100x100 mm for mammography
        roiSizeAUC = 100
        roiSizeBUC = 100
    else:
        # IEC 62220-1-1:2015 general radiography defaults
        default_A = 100  # width (mm)
        default_B = 50   # height (mm)
        _roiA = roiSizeA if roiSizeA is not None else default_A
        _roiB = roiSizeB if roiSizeB is not None else default_B
        # Non-uniformity correction ROI: 1.5x the largest MTF ROI dimension
        roiSizeAUC = int(max(_roiA, _roiB) * 1.5)
        roiSizeBUC = int(max(_roiA, _roiB) * 1.5)

    return {
        'roiSizeA': _roiA,
        'roiSizeB': _roiB,
        'roiSizeAUC': roiSizeAUC,
        'roiSizeBUC': roiSizeBUC,
    }


def _calculate_mammography_offset(doseImage, roiSizeAUC, pixelSpacing):
    """
    Calculate the X offset so the ROI is centred at 60 mm from the
    left edge of the image, as required for mammography (IEC 62220-1-1:2015).

    Parameters
    ----------
    doseImage : np.ndarray
        The linearised dose image (2D).
    roiSizeAUC : int
        Width of the non-uniformity correction ROI (mm).
    pixelSpacing : float
        Pixel spacing (mm/px).

    Returns
    -------
    int
        Offset in pixels from the image centre (for cropImage).
    """
    roiSizeUCPx = int(roiSizeAUC / pixelSpacing)
    imageWidth = doseImage.shape[1]
    desiredLeftEdge = int((60 - roiSizeAUC / 2) / pixelSpacing)
    targetCenterX = desiredLeftEdge + roiSizeUCPx // 2
    imageCenterX = imageWidth // 2
    return targetCenterX - imageCenterX


def calculateMTF(path, conversion, a, b, exportFormat,
                 progress_callback=None, roiSizeA=None, roiSizeB=None):
    """
    Calculate the Modulation Transfer Function (MTF).

    The imaging modality is automatically detected from the DICOM tag
    Modality (0008,0060). When mammography ('MG') is detected, the
    algorithm uses mammography-specific ROI sizes, non-uniformity
    correction parameters and ROI positioning (60 mm from the left edge).

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
    roiSizeA : int or None, optional
        ROI dimension A in mm. None = modality default.
        General: 100 mm (width), Mammography: 50 mm (height).
    roiSizeB : int or None, optional
        ROI dimension B in mm. None = modality default.
        General: 50 mm (height), Mammography: 25 mm (width).

    Returns
    -------
    dict
        Dictionary containing:
        - 'dataframe': pd.DataFrame with MTF results
        - 'rois': dict mapping file paths to (x, y, width, height)
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
    params = _get_modality_params(modality, roiSizeA, roiSizeB)
    roiSizeA = params['roiSizeA']
    roiSizeB = params['roiSizeB']
    roiSizeAUC = params['roiSizeAUC']
    roiSizeBUC = params['roiSizeBUC']

    print(f"MTF ROI: {roiSizeA} x {roiSizeB} mm")
    print(f"Non-uniformity correction ROI: {roiSizeAUC} x {roiSizeBUC} mm")

    # Flags to determine if both orientations have been obtained
    verticalFlag = 0
    horizontalFlag = 0

    # Offset (px) from center to trim ROI
    # For general: always (0, 0) — centred
    # For mammography: calculated per image (60 mm from left edge)
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

        # --- Mammography-specific offset ---
        if modality == 'MG':
            offsetCenterX = _calculate_mammography_offset(doseImage, roiSizeAUC, pixelSpacing)
            offsetCenterY = 0
            print(f"Mammography offset: X = {offsetCenterX} px (60 mm from left edge)")

        # -------------------------------------
        # Section 2*. Non-uniformity correction
        # -------------------------------------
        # Crop the ROI for the correction
        croppedImage, _, _, crop_start_row, crop_start_col = ReX.cropImage(
            doseImage, roiSizeBUC, roiSizeAUC,
            pixelSpacing, pixelSpacing,
            offsetCenterX, offsetCenterY
        )

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
        # For mammography the sub-ROI is cropped from the centre of the
        # correction ROI, so the offset is (0, 0) relative to corrImage.
        # For general radiography the original offset is reused.
        if modality == 'MG':
            mtf_offsetX = 0
            mtf_offsetY = 0
        else:
            mtf_offsetX = offsetCenterX
            mtf_offsetY = offsetCenterY

        # Crop roiSizeA x roiSizeB central ROI for MTF
        croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(
            corrImage, roiSizeB, roiSizeA,
            pixelSpacing, pixelSpacing,
            mtf_offsetX, mtf_offsetY
        )
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

            croppedROI, _, _, roi_start_row, roi_start_col = ReX.cropImage(
                corrImage, roiSizeA, roiSizeB,
                pixelSpacing, pixelSpacing,
                mtf_offsetX, mtf_offsetY
            )
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
            startCol = (k - 1) * N
            endCol = startCol + N

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
        start = 499  # Max(N * k) = 500 -> 500 - 1 (python index) = 499
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
        fNyq = 1 / (2 * pixelSpacing)

        # Calculate the FFT from LSF
        MTF = np.abs(np.fft.fft(LSF))

        # Take the positive half
        MTF = MTF[:len(MTF) // 2]

        # Normalize MTF
        MTF = MTF / MTF[0]

        # Calculate LSF length to create frequency vector
        LSFlength = len(LSF)

        # Frequency "step" between positions in the array
        frequencySpacing = 1 / (LSFlength * pixelSpacing * N)

        # Create the frequency vector
        frequencies = np.linspace(
            -0.5 * N / pixelSpacing,
            0.5 * N / pixelSpacing - frequencySpacing,
            LSFlength
        )

        # Take the positive half
        frequencies = np.fft.fftshift(frequencies)
        frequencies = frequencies[:len(frequencies) // 2]

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
            mask = (frequencies >= lowerLimit) & (frequencies <= upperLimit)

            # Average the values for the selected spatial frequencies
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
        ReX.exportData(
            frequencies, MTF_smoothed, MTF_smoothed,
            ['Frequencies (1/mm)', 'MTF', ''],
            path, name, exportFormat
        )

        i = i + 1

    # --------------------------
    # Section 9. Combine MTFs
    # --------------------------
    # If MTF has been calculated for vertical and horizontal orientation
    # it creates a file with the values prepared to calculate DQE later

    # Define target frequencies
    print("Defining Target frequencies.")
    sample_step = 0.1  # Frequency step (1/mm) añadido por AOL
    target_frequencies = np.arange(sample_step, fNyq + sample_step, sample_step)
    target_frequencies = target_frequencies[target_frequencies <= fNyq]
    print("Target frequencies defined.")

    # Initialize results dictionary
    results = {freq: {} for freq in target_frequencies}
    print("Result dictionary initialized")

    # Process each file in the directory
    print("Processing exported MTF files...")
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if "MTF_vertical" in file_name:
            print(f"  Found: {file_name}")
            data = ReX.process_file(file_path, target_frequencies, 'MTF', exportFormat)
            for freq, value in data.items():
                results[freq]['MTF Vertical'] = value
        elif "MTF_horizontal" in file_name:
            print(f"  Found: {file_name}")
            data = ReX.process_file(file_path, target_frequencies, 'MTF', exportFormat)
            for freq, value in data.items():
                results[freq]['MTF Horizontal'] = value

    # Create DataFrame from results
    final_df = pd.DataFrame(results).T
    final_df.index.name = 'Frequencies (1/mm)'

    column_order = ['MTF Horizontal', 'MTF Vertical']
    final_df = final_df[[col for col in column_order if col in final_df.columns]]

    if verticalFlag == 1 and horizontalFlag == 1:
        # Save final DataFrame
        if exportFormat == 'excel':
            output_file_path = os.path.join(path, 'MTF_to_DQE.xlsx')
            final_df.to_excel(output_file_path)
        elif exportFormat == 'csv':
            output_file_path = os.path.join(path, 'MTF_to_DQE.csv')
            final_df.to_csv(output_file_path)  # Fixed: was .to_excel() for csv

        print(f'Data saved as: {output_file_path}')

    progress_callback(100)
    progress_callback(0)
    return {'dataframe': final_df, 'rois': rois, 'modality': modality}