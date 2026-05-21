# ARCHIVO DE EJEMPLO. PENDIENTE DE IMPLEMENTACIÓN.

import numpy as np
import spekpy as sp

try:
    from matplotlib import pyplot as plt
except:
    print("Cannot import matplotlib. Please check that it is installed")
print("\n** Script to generate a filtered spectrum and plot using matplotlib **\n")

# Tube specifications
anode_angle_L = 16
anode_angle_s = 10
ventana_Be = 0.63  # en mm
anodo_mat = 'W'
Filtro = 'Rh'
Filtro_esp = 0.05  # en mm
Filtro_add = 'Al'
Filtro_add_esp = 2 # en mm

DFD = 65  # en cm

esp_air = DFD * 10 - ventana_Be - Filtro_esp
esp_air_add = DFD * 10 - ventana_Be - Filtro_esp - Filtro_add_esp

# Generate unfiltered spectrum
s_org = sp.Spek(kvp=28, th=anode_angle_L, dk=0.1, targ=anodo_mat, physics='kqp', mu_data_source='nist', z=DFD)
# Filter the spectrum
s_org.filter('Be', ventana_Be).filter(Filtro, Filtro_esp)
s = sp.Spek.clone(s_org).filter('Air', esp_air)
t = sp.Spek.clone(s_org).filter(Filtro_add, Filtro_add_esp).filter('Air', esp_air_add)
# s.set(ref_kerma=1)
s.summarize()
kerma = s.get_kerma()
emean = s.get_emean()
eeff = s.get_eeff()
hvl1 = s.get_hvl1()
hvl2 = s.get_hvl2()
hc = s.get_hc()
fluence = s.get_flu()
eflu = s.get_eflu()

flu_norm = s.get_norm_flu()

print('KERMA(µGy)', kerma, 'Emean(keV) ', emean, 'Eeff', eeff, 'HVL1 (mmAl)', hvl1, 'HVL2 (mmAl)', hvl2, 'HC ',
      hc, 'Fluencia', fluence, 'E Fluence ', eflu, 'Fluence normaliza', flu_norm)
# t.set(ref_kerma=1)
# Get energy values array and fluence arrays (return values at bin-edges)

karr_s, spkarr_s = s.get_spectrum(edges=True)
karr_t, spkarr_t = t.get_spectrum(edges=True)
# Plot spectrum
plt.plot(karr_s, spkarr_s)
plt.plot(karr_t, spkarr_t)
plt.xlabel('Energy [keV]')
plt.ylabel('Fluence per mAs per unit energy [photons/cm2/mAs/keV]')
plt.title('An example x-ray spectrum')
plt.show()

kerma_t = t.get_kerma()

snr_in2_s = np.trapz(spkarr_s, karr_s) / (100 * kerma)
snr_in2_t = np.trapz(spkarr_t,karr_t)/(100 *  kerma_t)
rend = kerma * np.power(DFD / 100.0, 2)
rend_t = kerma_t * np.power(DFD / 100.0, 2)

print('SNRin2 RQR-M2: ', snr_in2_s, 'Rendimiento @1m', rend, 'SNRin2 RQA-M2: ',snr_in2_t, 'Rendimiento @1m RQA: ', kerma_t)

# # Datos del espectro
# energia = np.array([1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75, 22.25, 22.75, 23.25, 23.75, 24.25, 24.75, 25.25, 25.75, 26.25, 26.75, 27.25, 27.75])
# fluencia = np.array([8.891621353659197e-304, 0, 0, 2.8070316639283004e-247, 2.931598372064195e-191, 5.890886307502309e-133, 1.0588334268066746e-92, 1.4174617315224389e-66, 7.438020638594368e-49, 2.0467328930589645e-36, 2.0214119305346575e-27, 9.05480614344613e-21, 1.3286400262708435e-15, 7.090691071228691e-12, 6.283793682635494e-8, 1.9698269323794217e-6, 0.00015122474433597956, 0.03572162316359436, 0.09924476721852483, 0.9669488748520461, 12.868327127641203, 43.17847544696406, 152.9338046946646, 503.5888260829074, 1437.1226623649413, 3529.893609427788, 7645.51051013403, 14897.045014348418, 26527.049432988642, 43753.5598380379, 67629.93835731997, 98808.69282776839, 137530.56840862895, 183257.14292719722, 235052.58460740934, 291834.7690736861, 351973.12585129973, 413591.38812745287, 474328.5381849554, 532085.8261980506, 585198.2778147603, 631773.6959557612, 670100.7302984907, 698539.8136252195, 27688.36567018342, 30684.378582843587, 35493.71935060319, 39765.81632013202, 42896.756490437045, 44221.94502538906, 42962.66288450863, 38207.7652729951, 28990.963714365455, 14454.095587380594])
#
# # Calcular la fluencia total utilizando la regla trapezoidal
# fluencia_total = np.trapz(fluencia, energia)
#
# print(f"Fluencia total de fotones: {fluencia_total/(4.7*100)} fotones/mm²")
