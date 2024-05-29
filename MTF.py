# ------------------------------------------------------ #
#           MODULATION TRANSFER FUNCTION (MTF)           #
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
import ReXfunc
import ReXfunc as ReX

# ---------------------
# Section 1. Parameters
# ---------------------
path = 'C:/Users/rafa_/OneDrive/Residencia/01. R1/02. Radiodiagnóstico/Aplicaciones/Cálculo MTF/ImagenesMTF/SanJose/Mesa/V/'
files = glob.glob(os.path.join(path, '*.DCM'))

# Export format (saved in path): 'excel' or 'csv'
exportFormat = 'excel'

# ROI size (mm) (IEC 62220-1-1:2015) considering vertical position of the edge
roiSizeA = 100  # width
roiSizeB = 50   # height

# ROI size (mm) for non-uniformity correction (IEC 62220-1-1:2015)
# It should be at least 1.5 times bigger than the other one
roiSizeAUC = 150
roiSizeBUC = 150

# Coefficients of the response function according to:
# PV = a·K + b (if linear)
# PV = a·ln(K) + b (if log)
conversion = 'linear' # or 'log'
a = 607.47
b = 67.70

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
i = 0
for file in files:

    # Read dcm file
    dicomFile = pydicom.dcmread(file)
    #dicomImage = dicomFile.pixel_array

    dicomImage = np.fliplr(np.flipud(dicomFile.pixel_array))


    plt.figure()  # Creates a new figure window
    plt.imshow(dicomImage, cmap='gray')  # Display the original image
    plt.colorbar()  # Display a color bar
    plt.title('Original Image')  # Title of the figure
    plt.show()

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

    # Substitute negative values -> zeros
    doseImage[doseImage < 0] = 0

    # -------------------------------------
    # Section 2*. Non-uniformity correction
    # -------------------------------------
    # Crop the ROI for the correction
    croppedImage, _, _ = ReX.cropImage(doseImage, roiSizeBUC, roiSizeAUC, pixelSpacing, pixelSpacing,
                                                        offsetCenterX, offsetCenterY)

    if isNeeded == 1:
        # Apply non-uniformity correction
        # Calculation of 2D best fit (S)
        # Corrected image = Original Image / S * S_avg
        corrImage = ReXfunc.correctedImage(croppedImage)

    elif isNeeded == 0:
        # Not needed
        corrImage = croppedImage

    # -------------------
    # Section 3. MTF ROI
    # -------------------
    # Crop 100 mm x 50 mm central ROI for MTF
    croppedROI, _, _ = ReX.cropImage(corrImage, roiSizeB, roiSizeA, pixelSpacing, pixelSpacing, offsetCenterX, offsetCenterY)

    # plt.figure()  # Creates a new figure window
    # plt.imshow(croppedROI, cmap='gray')  # Display the original image
    # plt.colorbar()  # Display a color bar
    # plt.title('MTF ROI')  # Title of the figure
    # plt.show()


    # -------------------------
    # Section 4. Edge Detection
    # -------------------------
    # Function to calculate angle and orientation
    angle, orientation = ReX.edgeDetection(croppedROI)

    absAngle = abs(angle)

    # Adjust image to horizontal orientation for next steps
    if orientation == 'vertical':

        NAngle = abs(90 - absAngle)
        croppedROI, _, _ = ReX.cropImage(corrImage, roiSizeA, roiSizeB, pixelSpacing, pixelSpacing, offsetCenterX,
                                         offsetCenterY)

        if absAngle > 90:
            croppedROI = np.rot90(croppedROI)
        elif absAngle < 90:
            croppedROI = np.flipud(np.rot90(croppedROI))

        plt.figure()  # Creates a new figure window
        plt.imshow(croppedROI, cmap='gray')  # Display the original image
        plt.colorbar()  # Display a color bar
        plt.title('Reoriented')  # Title of the figure
        plt.show()

    elif orientation == 'horizontal':
        NAngle = absAngle

    if angle > 0: # TBR: TEST!!
        croppedROI = np.fliplr(croppedROI)

    print(NAngle)

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

    #plt.figure()

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

        # Adds the current ESF to the plot
        # plt.plot(current_ESF, label=f'ESF for Group { k - 1}')

    # Set title and legend
    # plt.title('Superimposed ESFs')
    # plt.xlabel('Pixel')
    # plt.ylabel('Intensity')
    # plt.legend()  # Muestra una leyenda con las etiquetas de cada grupo

    # Show the graph
    # plt.show()

    # Calculation of average ESF
    average_ESF = np.mean(all_ESFs, axis=1)

    # Slice the ESF to avoid unuseful values due to displacement (not IEC)
    start = 499     # Max(N * k) = 500 -> 500 - 1 (python index) = 499
    average_ESF = average_ESF[start:]

    plt.figure()
    plt.plot(average_ESF)
    plt.title('Average ESF (oversampled)')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    plt.show()

    # -------------------------------------
    # Section 6. Line Spread Function (LSF)
    # -------------------------------------
    # Define kernel for convolution
    kernel = np.array([-1, 0, 1])

    # Convolution
    LSF = np.convolve(average_ESF, kernel, mode='valid')

    # Plot LSF
    # plt.figure()
    # plt.plot(LSF)
    # plt.title('Line Spread Function (LSF)')
    # plt.xlabel('Pixel position')
    # plt.ylabel('Y')
    # plt.show()

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

    # Plot MTF
    plt.figure()
    plt.plot(MTF)
    plt.title('Modulation Transfer Function (MTF)')
    plt.xlabel('Frequency (1/mm)')
    plt.ylabel('MTF')
    plt.show()

    # ------------------------------
    # Section 7.b. Smoothing the MTF
    # ------------------------------
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

    # Plot MTF
    plt.figure()
    plt.plot(MTF_smoothed)
    plt.title('Smoothed Modulation Transfer Function (MTF)')
    plt.xlabel('Frequency (1/mm)')
    plt.ylabel('MTF')
    plt.show()

    # --------------------------
    # Section 8. Export MTF data
    # --------------------------
    # Slice frequencies below Nyquist frequency
    validIndexes = frequencies <= fNyq
    frequencies = frequencies[validIndexes]
    MTF_smoothed = MTF_smoothed[validIndexes]

    # Plot MTF
    # Crear el plot
    plt.figure()
    plt.plot(frequencies, MTF_smoothed, marker='o', linestyle='-')  # Puedes ajustar el estilo de línea y marcador

    # Configurar las etiquetas y títulos
    plt.xlabel('Frecuencia (1/mm)')
    plt.ylabel('MTF')
    plt.title('Modulation Transfer Function Smoothed')
    plt.show()

    # Actual export
    name = 'MTF_' + orientation
    ReX.exportData(frequencies, MTF_smoothed, MTF_smoothed, ['Frequencies (1/mm)', name, ''], path, name, exportFormat)

    i = i + 1