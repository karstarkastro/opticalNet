import matplotlib.pyplot as plt
import numpy as np

###
#   @brief Plots the Stokes vector in 3D space and 2D planes
#
#   Plots the Poincaré Sphere and the 2D Stokes planes 
#	where the 2 axis are S1,S3 and S2,S3
#
#   @param signal 	Instance of class opticalSignalTransmission
#
#   @return None
###
def stokesRepresentation(signal):
	xPol = signal[0,:]
	yPol = signal[1,:]
	S0 = np.square(abs(xPol)) + np.square(abs(yPol))
	S1 = np.square(abs(xPol)) - np.square(abs(yPol))
	S1 = np.divide(S1, S0)
	S2 = 2*abs(xPol)*abs(yPol)*np.cos(np.angle(yPol) - np.angle(xPol))
	S2 = np.divide(S2, S0)
	S3 = 2*abs(xPol)*abs(yPol)*np.sin(np.angle(yPol) - np.angle(xPol))
	S3 = np.divide(S3, S0)

	ax = plt.axes(projection = '3d')
	ax.scatter(S1, S2, S3)
	plt.xlim(-1,1)
	ax.set_title('Poincaré Sphere ' + signal.mod)
	plt.show()

	plt.scatter(S1, S3, color='blue')
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	plt.xlabel('S1')
	plt.ylabel('S3')
	plt.title('2D Stokes Plane (S1,S3) ' + signal.mod)
	plt.show()

	plt.scatter(S2, S3, color='red')
	plt.xlabel('S2')
	plt.ylabel('S3')
	plt.title('2D Stokes Plane (S2,S3) ' + signal.mod)
	plt.show()

###
#   @brief Calculates the BER
#
#   Calculates the bit error rate of a received sequence 
#
#   @param dataOUt	ndarray of the received binary data
#   @param dataIn	ndarray of the transmitted binary data
#	@param npol		Int representing number of polarizations of the optical signal
#	@param snr 		Int representing the Signal to Noise ratio [dB] of the rx signal 
#
#   @return None
###
def BER(dataOut, dataIn, npol, snr): 
	pol = ('X','Y')
	for i in range(npol):
		numErr = np.bitwise_xor(dataOut[i,:], dataIn[i,:]).sum()
		ber = numErr/len(dataIn[i,:])
		BER = "{:.2e}".format(ber)	# BER in scientific notation
		if npol > 1:
			print(pol[i] + " polarization")
		print("For a SnR of " + str(snr) + "dB the obtained BER is " + str(BER) + ", for " \
			+ str(numErr) + " errors")

###
#   @brief Plots the constellation diagram of the received signal
#
#   Plots the constellation diagram of the received quadrature signal
#	in yellow while the expected symbols are plotted in blue
#
#   @param signal 	Instance of class opticalSignalTransmission
#
#   @return None
###
def constellationDiag(signal):
	if signal.npol == 2:
		titles = ('X polarization Rx', 'Y polarization Rx')
		for i in range(signal.npol):
			plt.subplot(1, 2, i + 1)
			plt.scatter(signal[i,:].real, signal[i,:].imag, color='yellow', \
			 label='Received symbol')
			plt.scatter(signal.alphabet.real, signal.alphabet.imag, color='blue', label='Expected symbol', marker='x', s=75)
			plt.xlabel('real')
			plt.ylabel('imaginary')
			plt.title(titles[i] + ' (' + signal.mod + ')')
			plt.legend(loc='upper right')
	else:
		plt.scatter(signal[0,:].real, signal[0,:].imag, color='yellow', label='Received symbol')
		plt.scatter(signal.alphabet.real, signal.alphabet.imag, color='blue', label='Expected symbol', marker='x', s=75)
		plt.xlabel('real')
		plt.ylabel('imaginary')
		plt.title('Constellation Diagram ' + signal.mod)
		plt.legend(loc='upper right')
	plt.show()
