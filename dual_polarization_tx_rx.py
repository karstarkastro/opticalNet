'''
Transmission and Reception of an optical signal

DETAILS

* Modulation options: 4-PSK, 8-PSK, 4-QAM, 16-QAM, 64-QAM, 128-QAM, 256-QAM
* Polarization: Single/Dual
* Pulse shaping using RRC filter
* Channel Effects:
	- Rotation of state of polarization
	- Phase noise
	- AWGN
	- Polarization Mode Dispersion
	- Chromatic Dispersion


USAGE: python fileName.py [-m modulation][-sps samples_per_symbol] 
		[-snr signal_to_noise_ratio_rx] [-L length] [-wlength wavelength] 
		[-pmd differential_group_delay] [-cd chromatic_dispersion] -pn -dp

OPTIONS:
		-m 		modulation
				Specify desired modulation format: 4-QAM, 16-QAM (default if omitted), 64-QAM or 256-QAM
		-sps 	samples_per_symbol 
				Number of samples per symbol. Must be an integer. Default = 4
		-snr 	signal_to_noise_ratio_rx
				Specify signal to noise ratio (dB) at receiver. Must be an integer. Default = 10 [dB]
		-pn 	phase_noise
				Adds phase noise to the signal
		-L		length
				Length of the SMF in kilometers. Default = 500 [km]
		-wlength wavelength
				 Wavelength of the signal in meters. Default =  1550[nm]
		-dp 	dualpolarization
				Transmission of a dually polarized optical signal affected by polarization rotation
				which is corrected by means of the CMA algorithm (demux applies only to 4-QAM)
		-pmd	polarization mode dispersion
				Differential Group Delay in seconds. It is the difference in propagation time between
				the vertical and horizontal polarization modes. Default = 10e-12[s]
		-cd		chromatic dispersion
				Dispersion induced by the dependence of the index of refraction of the fiber on the
				wavelength of the propagating signal. Measured in [s²/m]. Default = 17e-6[s²/m]
'''

import argparse
import numpy as np
from opticalNetwork import digCohRx, impairments, measurements, signals 


parser = argparse.ArgumentParser()
parser.add_argument("-m","--modulation", help = "QAM modulation type [4, 16, 64 or 256-QAM]", \
 default = "16-QAM")
parser.add_argument("-sps", help = "Number of samples per symbol", default = 4, type = int)
parser.add_argument("-snr", help = "Signal to Noise Ratio in dB", default = 10, type = int)
parser.add_argument("-a", "--attenuation", help = "Attenuation in dB", default = 0.2, type = float)
parser.add_argument("-pn", "--phasenoise", help = "Adds phase noise", action = "store_true")
parser.add_argument("-dp", "--dualpolarization", help = "Transmits a dually polarized signal", action = "store_true")
parser.add_argument("-pmd", help = "Polarization Mode Dispersion", default = 10e-12, type = float)
parser.add_argument("-cd", help = "Chromatic Dispersion", default = 17e-6, type = float)
parser.add_argument("-L", "--length", help = "Fiber Length", default = 500, type = float)
parser.add_argument("-wlength", help = "wavelength", default = 1550e-9, type = float)
parser.add_argument("-n", "--spans", help = "Number of spans", default = 2, type = int)
args = parser.parse_args()


mod = args.modulation.upper()
numSymbols = 2**16
numSpS = args.sps
baudRate = 28e9 	# symbols per second
fs = 2*baudRate	# sampling frequency
nmodes = 2 if args.dualpolarization else 1	# number of polarizations for Polarization Multiplexing

snrdB = args.snr
length = args.length*1e3 	# fiber length in meters
linewidth = 2e3	# linewidth of the laser source
theta = np.pi/7		# azimuth angle for polarization rotation
phi = np.pi/5		# elevation angle for polarization rotation

# include pilot signal if the modulation signal is 16-QAM or higher
pilot = True if (args.dualpolarization and (('PSK' or '4-QAM') not in mod)) else False

if pilot:
	sig1 = signals.opticalSignalwithPilotTx(mod, numSymbols, numSpS, baudRate, fs, snrdB, npol = nmodes)
else:
	sig1 = signals.opticalSignalTx(mod, nmodes, numSymbols, numSpS, baudRate, fs, snrdB)

noisySig = impairments.addOpticalImpairments(sig1, args.cd, args.wlength, length, args.phasenoise,\
 linewidth, args.dualpolarization, args.pmd, theta, phi)

# RECEPTION OF THE SIGNAL

dispCompensatedSig = digCohRx.dispersionComp(noisySig, args.cd, args.wlength, length)
rxFiltSignal = dispCompensatedSig.matchedFilt()

if pilot:
	rxPolDemuxSignal = digCohRx.polarizationDemuxPilotBased(rxFiltSignal)
else:
	rxPolDemuxSignal = digCohRx.polarizationDemux(rxFiltSignal)

# VISUALIZATION OF POINCARÉ SPHERE AND STOKES PLANE

if args.dualpolarization:
	measurements.stokesRepresentation(rxPolDemuxSignal)

# DATA DEMODULATION

demodulatedSignal = rxPolDemuxSignal.demodulation()

# BER CALCULATION 

measurements.BER(demodulatedSignal, rxPolDemuxSignal.dataIn, rxPolDemuxSignal.npol, snrdB)

# PLOT OF CONSTELLATION DIAGRAM

measurements.constellationDiag(rxPolDemuxSignal)


