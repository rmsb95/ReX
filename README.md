# ReX
#### Developed by: @rmsb95
#### Date: 06/2024
#### Version: 1.1
Apps developed for the calculation of NPS, NNPS, MTF & DQE parameters in the context of Radiodiagnostics Devices QA.
The calculations have been made following UNE-EN IEC 62220-1-1:2015, with some additional considerations when explicit
instructions where not given.

## 1. Codes
### 1.1 Main
The User Interface (UI) code is hosted here. It is the main python file that calls the other functions for calculation.
### 1.2. ReXfunc
In order to improve reading, some functions used in ReXMTF, ReXNNPS & ReXDQE have been hosted here.
### 1.3. ReXMTF
Contains functions related to the calculation of the Modulation Transfer Function (MTF).
### 1.4. ReXNNPS
Contains functions related to the calculation of the [Normalized] Noise Power Spectrum (NPS & NNPS).
### 1.5. ReXDQE
Contains functions related to the calculation of the Detective Quantum Efficiency (DQE).

## 2. Resources
### 2.1. Resources
List of dependencies needed to run the project:
pip install -r requirements.txt
### 2.2. Additional resources
Additional resources can be found in the 'resources' directory, mainly related to UI.

## 3. Instructions
### 3.1. Setup
1. Clone the repository: git clone github.com/rmsb95/ReX
2. Navigate to the project directory: cd projectPath
3. Install the dependencies: pip install -r requirements.txt
### 3.2. Running ReX
To run the application, execute the following command: python main.py
### 3.3. Using ReX
#### A. NPS, NNPS & MTF
1. Open the app.
2. Choose the directory containing the DICOM files.
- NPS & NNPS: the DICOM images should have been taken under the same exposure conditions.
- MTF: the DICOM images should have been taken under the same exposure condition. There should be at least two images: one in vertical disposition of the tungsten plate and another one in horizontal disposition.
3. Configure the parameters: converse function, export format.
4. Click the calculation: NPS & NNPS or MTF.
5. That's it! You'll get the results in the same path:
- NPS & NNPS: there will be three files: 'NPS_data', 'NNPS_data' and 'NNPS_to_DQE'
- MTF: there will be one file for each orientation: 'MTF_horizontal', 'MTF_vertical'; and additional file 'MTF_to_DQE' will be created if the previous two files exists.
#### B. DQE
1. Open the app.
2. Click the button: 'Calcular DQE' to open up the DQE calculation window.
3. Select the radiation quality used and input the air kerma measured in microGray.
4. Click the button 'Archivo MTF' and select the file 'MTF_to_DQE' created in A.
5. Click the button 'Archivo NNPS' and select the file 'NNPS_to_DQE' created in A.
6. Click on 'Calcular DQE'. A new window will appear to save the file.


## 4. Version control
### Version 1.0.1
Bug corrections:
- Added openpyxl import to every ReX file.

### Version 1.0.0
Main changes:
- MTF & NNPS values used to calculate DQE are obtained through a cubic spline fit to the data.

## 5. Contact
For any questions or issues, please contact the developer at rmsb95@gmail.com [Issue: ReX].