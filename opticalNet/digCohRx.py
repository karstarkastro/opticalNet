from opticalNet import constants
import numpy as np
from scipy.fftpack import fft, fftshift, ifft, ifftshift

###
#   @brief Chromatic Dispersion Compensation
#   
#   Chromatic Dispersion Compensation is achieved by doing the product
#   in the frequency domain between the received signal and the inverse
#   transfer function of the dispersion suffered by the optical signal
#   through the channel. 
#   
#   @param signal       2-by-n matrix where the rows are the X and Y polarizations
#   @param dispersion   Float representing the chromatic dispersion of the fiber in [s/m²]
#   @param h            Float representing the wavelength of the signal in [m]
#   @param length       Float representing the length of the fiber in [m]
#   @param baudRate     Int representing the number of symbols per second
#	@param npol			Int representing the number of polarizations sent through the fiber
#
#   @return 2-by-n matrix where the rows are the dispersion-compensated signals
# 
###
def dispersionComp(signal, dispersion, h, length):
	rxCompensated = np.empty(signal.shape, complex)
	for i in range(signal.npol):
		l = signal[i,:].size
		dt = 1/signal.baudRate
		timegrid = dt*np.arange(0,l+1)
		df = 1/(timegrid[-1] - timegrid[0])
		f = df*np.arange(-l/2, l/2)
		w = 2*np.pi*f
		beta2 = -(h**2)*dispersion/(2*np.pi*constants.c)
		PFin = fftshift(fft(signal[i,:]))
		TF_DispFiber  = np.exp(1j*beta2*w**2/2*length)
		PFout = ifft(ifftshift(PFin*TF_DispFiber))
		rxCompensated[i,:] = PFout
	return signal.convert_to_signal_class(rxCompensated)

###
#   @brief Polarization Demultiplexing by CMA algorithm
#   
#   Polarization Demux is accomplished by implementing the Constant Modulus
#   Algorithm [1]. This algorithm is used to demultiplex M-PSK modulations, where
#   the radius of the circle formed in the constellation diagram has unit value
#   in all cases. This algorithm is not valid for M-QAM modulations with the 
#   exception of 4-QAM which is basically QPSK.
#   
#   @param signal   2-by-n matrix where the rows are the X and Y polarizations
#
#   @return 2-by-n matrix where the rows are the demultiplexed signals
#
#   [1] Kikuchi K., (2011) "Performance Analysis of polarization demultiplexing based on
#   constant-modulus algorithm in digital coherent optical receivers" in Optics express 19
#   (10), 9868-9880
# 
###
def polarizationDemux(signal):
	cmaMu = 1/6000		# step-size parameter for polarization demux using CMA
	cmaTaps = 109		# number of taps for the CMA algorithm
	cmaRadius = [1,1]	# radius of CMA
	if signal.npol == 2:
		hxx_in = np.zeros(cmaTaps, complex)
		halfTaps = int(np.floor(cmaTaps/2))
		hxx_in[halfTaps] = 1
		hyy_in = hxx_in
		hxy_in = np.zeros(cmaTaps, complex)
		hyx_in = hxy_in
		c = 0
		convergence = False
		L = signal.shape[1]
		repetitions = int(50*np.ceil(1/(L*cmaMu)))
		while convergence != True and c < repetitions:
			c = c + 1
			x_out, y_out, hxx_out, hyy_out, hxy_out, hyx_out = cmaFilter(signal, cmaRadius, cmaTaps, \
	        	cmaMu, hxx_in, hyy_in, hxy_in, hyx_in)
			if np.concatenate((hxx_out - hxx_in, hyy_out - hyy_in, hxy_out - hxy_in,\
				hyx_out - hyx_in)).max() < 5e-5:
				convergence = True
			hxx_in = hxx_out
			hyy_in = hyy_out
			hxy_in = hxy_out
			hyx_in = hyx_out
		demuxSignal =  np.vstack((x_out, y_out))
		return signal.convert_to_signal_class(demuxSignal)
	else:
		return signal

###
#   @brief CMA algorithm
#   
#   Implementation of the Constant Modulus Algorithm.
#   
#   @param signal   2-by-n matrix where the rows are the X and Y polarizations and
#                   the columns are the symbols in their complex representation.
#   @param radius   The radius of the circles seen in the constellation diagram.
#                   The radius is always 1 for M-PSK signals.
#   @param taps     Number of taps of the CMA filter
#   @param mu       Step-size of the CMA algorithm
#   @param hxx      Adaptive filter to control the state of polarization of the X polarization
#   @param hxy      Adaptive filter to control the state of polarization of the X polarization
#   @param hyy      Adaptive filter to control the state of polarization of the Y polarization
#   @param hyx      Adaptive filter to control the state of polarization of the Y polarization

#
#   @return Array of n elements representing the output of the X polarization
#   @return Array of n elements representing the output of the Y polarization
#   @return Updated values of the adaptive filters to control the SoP of the X polarization (hxx and hxy)
#   @return Updated values of the adaptive filters to control the SoP of the Y polarization (hyy and hyx)
# 
###
def cmaFilter(signal, radius, taps, mu, hxx, hyy, hxy, hyx):
    signal = np.concatenate((np.zeros((2, int((taps-1)/2 if taps%2 != 0 else taps/2))), \
        signal, np.zeros((2, int((taps-1)/2 if taps%2 != 0 else taps/2)))), axis = 1) # zero-padding the signal
    L = signal.shape[1]
    x_new = np.zeros(L, complex)
    y_new = np.zeros(L, complex)
    err_x = np.zeros(L, complex)
    err_y = np.zeros(L, complex)
    for q in range(L - taps + 1):
        X = signal[0, taps+q-1:None if q == 0 else q-1:-1]
        Y = signal[1, taps+q-1:None if q == 0 else q-1:-1]
        x_new[q] = np.dot(hxx, X) + np.dot(hxy, Y)
        y_new[q] = np.dot(hyx, X) + np.dot(hyy, Y)
        err_x[q] = radius[0] - np.square(abs(x_new[q]))
        err_y[q] = radius[1] - np.square(abs(y_new[q]))
        hxx = hxx + mu*err_x[q]*x_new[q]*X.conj() 
        hxy = hxy + mu*err_x[q]*x_new[q]*Y.conj() 
        hyx = hyx + mu*err_y[q]*y_new[q]*X.conj() 
        hyy = hyy + mu*err_y[q]*y_new[q]*Y.conj() 
    x_new = x_new[1 if taps%2 == 0 else None:-taps+1] # discarding unnecessary data introduced by zero-padding
    y_new = y_new[1 if taps%2 == 0 else None:-taps+1]
    return x_new, y_new, hxx, hyy, hxy, hyx
