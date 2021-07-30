from opticalNet import constants
import math
import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifftshift


###
#   @brief Adds optical impairments to the optical signal
#
#	Adds the following channel impairments to the optical signal:
#		- Additive White Gaussian Noise 
#		- Chromatic Dispersion
#		- Polarization Mode Dispersion
#		- State of Polarization Rotation
#		- Phase Noise from the laser source
#
#   @param signal 		Instance of the class opticalSignalTransmission
#   @param cd 			Float representing the chromatic dispersion of the fiber in [s/m²]  
#   @param wavelength 	Float representing the wavelength of the signal in [m]
#   @param length 		Float representing the length of the fiber in [m]
#   @param pn 			Boolean specifying if phase noise should be included or not. Default = false
#   @param linewidth 	Int representing the linewidth of the laser source in Hz
#   @param dp 			Boolean specifying if dual polarization is enabled. Default = false
#   @param pmd 			Floatrepresenting the Differential Group Delay between the x and y 
#						polarizations in [s]. Default = 10e-12 [s]
#   @param theta 		Float representing the azimuth rotation angle
#   @param phi 			Float representing the elevation rotation angle
#
#   @return 1D array of signal with noise
###
def addOpticalImpairments(signal, cd, wavelength, length, pn, linewidth, dp, pmd, theta, phi):
	channelMatrix = np.array([[np.cos(theta), np.exp(-1j*phi)*np.sin(theta)],\
	[-np.exp(1j*phi)*np.sin(theta), np.cos(theta)]])	# Jones matrix of the optical channel
	rxSignal_noisy = np.empty((signal.npol, signal.shape[1]), dtype = complex)
	for i in range(signal.npol):
		# Adding AWGN
		rxSignal_noisy[i,:] = awgn(signal[i,:], signal.snr)
		# Chromatic Dispersion
		rxSignal_noisy[i,:] = addCD(rxSignal_noisy[i,:], cd, wavelength, length, signal.baudRate)
		# Adding phase noise
		if pn: 
			rxSignal_noisy[i,:] = addPhaseNoise(rxSignal_noisy[i,:], linewidth, signal.fs)
	if dp:
		# Adding polarization rotation
		rxSignal_noisy = addPolarizationRotation(rxSignal_noisy, channelMatrix)
		# Polarization Mode Dispersion
		rxSignal_noisy = addPMD(rxSignal_noisy, pmd, signal.fs)
	return signal.convert_to_signal_class(rxSignal_noisy)

###
#   @brief Adds AWGN noise to the signal
#
#   @param x Noise free complex baseband input signal
#   @param snr Signal to Noise Ratio in dB
#
#   @return 1D array of signal with noise
###
def awgn(x, snr):

    L = x.size
    snrLinear = 10**(snr/10)
    Esym = np.sum(np.square(np.abs(x)))/L   # calculating average symbol energy
    N0 = Esym/snrLinear # noise spectral density
    noiseSigma = math.sqrt(N0/2)     # standard deviation for the noise
    n = noiseSigma*(np.random.randn(len(x)) + 1j*np.random.randn(len(x)))                            
    
    return x + n

###
#   @brief Adds chromatic dispersion to the signal
#
#   Adds chromatic dispersion to the signal in the frequency domain
#
#   @param signal       2D array representing the transmitted signal. The first row
#                       is the x polarization while the second row is the y polarization
#   @param dispersion   Float representing the chromatic dispersion of the fiber in [s/m²]                   
#   @param h            Float representing the wavelength of the signal in [m]
#   @param length       Float representing the length of the fiber in [m]
#   @param baudRate     Int representing the number of symbols per second
#
#   @return 2D array representing the signal with chromatic dispersion 
#
###
def addCD(signal, dispersion, h, length, baudRate):
    l = signal.size
    dt = 1/baudRate
    timegrid = dt*np.arange(0,l+1)
    df = 1/(timegrid[-1] - timegrid[0])
    f = df*np.arange(-l/2, l/2)
    w = 2*np.pi*f
    beta2 = -(h**2)*dispersion/(2*np.pi*constants.c)
    PFin = fftshift(fft(signal))
    TF_DispFiber  = np.exp(-1j*beta2*w**2/2*length)
    PFout = ifft(ifftshift(PFin*TF_DispFiber))
    return PFout

###
#   @brief Adds Phase noise to the signal
#
#   Calculates phase noise based on a Wiener noise process, with
#   variance given by sigma**2 = 2*pi*linewidth/fs
#
#   @param signal 1D array representing the transmitted signal
#   @param linewidth Linewidth of the laser source in Hz
#   @param fs Sampling frequency in Hz
#
#   @return 1D array representing the signal with the added phase noise
###
def addPhaseNoise(signal, linewidth, fs):

    var = 2*np.pi*linewidth/fs
    f = np.random.normal(scale = np.sqrt(var), size = signal.shape)
    f[0] = 0    # characteristic of a Wiener process
    ph = np.cumsum(f)
    index = np.shape(ph)[0]
    for nnoise in range(index):
        ph[nnoise] = ph[nnoise] - (nnoise/(index - 1))*ph[-1]
    return signal*np.exp(1j*ph)

###
#   @brief Adds polarization rotation to the signal
#
#   Performs the polarization rotation of a dual-polarized signal 
#   propagating through an optical channel 
#
#   @param signal 2D array representing the transmitted signal. The first row
#                 is the x polarization while the second row is the y polarization
#   @param channel 2D array representing the Jones matrix of the optical channel
#
#   @return 2D array representing the signal with polarization rotation
###
def addPolarizationRotation(signal, channel):
    return np.dot(channel, signal)

###
#   @brief Adds Polarization Mode Dispersion to the signal
#
#   Adds differential group delay to a dual-polarized signal 
#   propagating through an optical channel. This function is
#   heavily based on the one used by QAMPy library from Chalmers
#   Photonics Lab[1]
#
#   @param signal 2D array representing the transmitted signal. The first row
#                 is the x polarization while the second row is the y polarization
#   @param dgd    Floating point number representing the Differential Group
#                 Delay between the x and y polarizations 
#   @param fs     Int representing the sampling frequency
#
#   @return 2D array representing the signal with PMD
#
#   [1] Jochen Schröder and Mikael Mazur, "QAMPy a DSP chain for optical communications", 
#       DOI: 10.5281/zenodo.1195720
###
def addPMD(signal, dgd, fs):
    omega = 2*np.pi*np.linspace(-fs/2, fs/2, signal.shape[1], endpoint = False)
    Sf = fftshift(fft(ifftshift(signal, axes = 1), axis = 1), axes = 1)
    # h = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Sff = np.dot(h, Sf)
    h2 = np.array([np.exp(-1j*omega*dgd/2), np.exp(1j*omega*dgd/2)])
    Sn = Sf*h2
    # h = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    # Sf2 = np.dot(h, Sn)
    SS = fftshift(ifft(ifftshift(Sn, axes = 1), axis = 1), axes = 1)
    return SS.astype(complex)
