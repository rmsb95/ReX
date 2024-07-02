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
import pandas as pd
from PyQt5.QtWidgets import QFileDialog


def calculateDQE(nnps_file, mtf_file, beamQuality, kermaAir):

    # Select proper SNR^2 value in [1/(mm^2 * microGy)] from Table 3
    if beamQuality == 'RQA3':
        SNR2 = 20673
    elif beamQuality == 'RQA5':
        SNR2 = 29653
    elif beamQuality == 'RQA7':
        SNR2 = 32490
    elif beamQuality == 'RQA9':
        SNR2 = 31077

    # Calculating the Noise Power Spectrum of the radiation field at the detector surface
    W_entrada = kermaAir * SNR2

    # Reading NNPS & MTF data
    nnps_data = pd.read_excel(nnps_file)
    mtf_data = pd.read_excel(mtf_file)

    # Rename frequency columns in nnps_data to make it merge later
    nnps_data = nnps_data.rename(columns={nnps_data.columns[0]: mtf_data.columns[0]})

    # Asegurar que ambos dataframes están alineados por la columna de frecuencia
    merged_data = pd.merge(mtf_data, nnps_data, on=mtf_data.columns[0])

    # Calcular los nuevos valores
    merged_data['DQE vertical'] = (merged_data['MTF vertical'] ** 2 / W_entrada) / merged_data['NNPS Vertical']
    merged_data['DQE horizontal'] = (merged_data['MTF horizontal'] ** 2 / W_entrada) / merged_data[
        'NNPS Horizontal']

    return merged_data