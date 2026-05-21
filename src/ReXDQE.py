# ------------------------------------------------------ #
#           DETECTIVE QUANTUM EFFICIENCY (DQE)           #
# ------------------------------------------------------ #
# Developer: Rafael Manuel Segovia Brome
# Date: 05-2024
# Version: 1.1
#
# ---------------------
# Section 0. Imports
# ---------------------
import os
import openpyxl
import pandas as pd

# ---------------------
# Section 1. Constants
# ---------------------

# SNR^2 values in [mm^-2 · µGy^-1] from IEC 62220-1-1:2015, Table 3.
# Centralised here so that both the calculation function and the UI
# can share the same data without duplication.
SNR2_TABLE = {
    'RQA3':         20673,
    'RQA5':         29653,
    'RQA7':         32490,
    'RQA9':         31077,
    'RQA-M1 Mo/Mo': 4639,
    'RQA-M2 Mo/Mo': 4981,
    'RQA-M3 Mo/Mo': 5303,
    'RQA-M4 Mo/Mo': 6325,
    'Mo/Rh 28kV':   5439,
    'Rh/Rh 28kV':   5944,
    'W/Rh 28kV':    5975,
    'W/Al 28kV':    6575,
    'W/Rh 29kV':    6340
}

# ---------------------
# Section 2. Functions
# ---------------------

def calculateDQE(nnps_file, mtf_file, beamQuality, kermaAir):

    # Select proper SNR^2 value in [mm^-2 · µGy^-1] from SNR2_TABLE
    if beamQuality not in SNR2_TABLE:
        raise ValueError(f"Calidad de haz no reconocida: '{beamQuality}'. "
                         f"Valores válidos: {list(SNR2_TABLE.keys())}")
    SNR2 = SNR2_TABLE[beamQuality]

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
    print("To calculate new values")
    try:
        merged_data['DQE horizontal'] = (merged_data['MTF Horizontal'] ** 2 / W_entrada) / merged_data['NNPS Horizontal']
        merged_data['DQE vertical'] = (merged_data['MTF Vertical'] ** 2 / W_entrada) / merged_data['NNPS Vertical']
        return merged_data
    except Exception as e:
        print(f"Error processing: {e}")
        return None