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

### 1.4. ReXNNPS

### 1.5. ReXDQE

## 2. Resources

## 3. Instructions

## 4. Version control
### Version 1.1
Main changes:
- MTF & NNPS values used to calculate DQE are obtained through a 3 degree polynomial fit to the data.
