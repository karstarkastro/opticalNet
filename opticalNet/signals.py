import numpy as np
from scipy.signal import upfirdn



class opticalSignalTransmission(np.ndarray):

	_inheritattr_ = ["_mod", "_npol", "_numSpS", "_baudRate", "_fs", "_alphabet", "_snr", "_dataIn", "_numBits"]

	constellations = {
	'4-PSK': 4,
	'8-PSK': 8,
	'4-QAM': 4,
	'16-QAM': 16,
	'64-QAM': 64,
	'256-QAM': 256
	}

	rrcSpan = 10 		# number of taps of the filter
	rrcRolloff = 0.25

	def __new__(cls, mod, npol, numSymbols, numSpS, baudRate, fs, snr, filter = "rrc"):
		M = cls.constellations[mod]
		k = int(np.log2(M))		# number of bits per symbol
		numbits = int(numSymbols*k)

		# GENERATION OF RANDOM BITS OF DATA
		'''
		The first row corresponds to the x polarization while
		the second row corresponds to the y polarization
		'''
		dataIn = np.random.randint(2, size = (npol, numbits))
		'''
		the sequence of bits which are contained in a 1D array
		are transformed to a 2D matrix of dimensions:
			- numbits/k rows
			- k columns
		'''
		dataInMatrix = np.reshape(dataIn, (npol, int(numbits/k), k))
		dataSymbolsIn = np.empty((npol, int(numbits/k)), dtype = int)
		for i in range(npol):
			dataSymbolsIn[i,:] = cls.bi2de(dataInMatrix[i, :, :])	# converts binary data to decimal integers

		# PERFORM THE MODULATION AND PULSE SHAPING

		dataMod = np.empty((npol, int(numbits/k)), dtype = complex)
		for i in range(npol):
			if 'QAM' in mod:
				dataMod[i, :], alphabet = cls.qam_modulate(dataSymbolsIn[i, :], M)
			else:
				dataMod[i, :], alphabet = cls.psk_modulate(dataSymbolsIn[i, :], M)
		# the modulated signal goes through upsampling and pulse shaping
		'''
		The upfirdn implementation in SciPy follows the same logic as in Matlab's
		making the length of the resulting output of the filter equal to
		[(length(input_signal) - 1) * upsampling_factor + length(filter)] / downsampling_factor
		'''
		if filter == "rrc":
			signalFilter = cls.rrcosine(cls.rrcRolloff, cls.rrcSpan, numSpS)
		txSignal = np.empty((npol, ((dataMod.shape[1] - 1)*numSpS + signalFilter.size)), dtype = complex)
		for i in range(npol):
			txSignal[i,:] = upfirdn(signalFilter, dataMod[i,:], numSpS, 1)
		obj = txSignal.view(cls)
		obj._mod = mod
		obj._M = M
		obj._npol = npol 
		obj._numSpS = numSpS
		obj._baudRate = baudRate
		obj._fs = fs
		obj._alphabet = alphabet
		obj._snr = snr
		obj._dataIn = dataIn
		obj._numBits = numSymbols*k
		return obj

	###
	#   @brief Converts a numpy.ndarray to an instance of type OpticalSignalTransmission
	#
	#   @param arr An instance of type numpy.ndarray representing an optical signal
	#
	#   @return obj An instance of OpticalSignalTransmission 
	###
	def convert_to_signal_class(self, arr):
		obj = arr.view(self.__class__)
		for attr in self._inheritattr_:
			setattr(obj, attr, getattr(self, attr))
		return obj

	@property
	def npol(self):
		return self._npol

	@property
	def baudRate(self):
		return self._baudRate

	@property
	def mod(self):
		return self._mod

	@property
	def M(self):
		return self._M
	

	@property
	def numSpS(self):
		return self._numSpS

	@property
	def fs(self):
		return self._fs

	@property
	def alphabet(self):
		return self._alphabet

	@property
	def snr(self):
		return self._snr

	@property
	def dataIn(self):
		return self._dataIn
	
	@property
	def numBits(self):
		return self._numBits	

	###
	#   @brief Converts a binary matrix to a row of non-negative decimal integers
	#
	#   @param b Matrix with binary elements
	#
	#   @return 1D array of positive decimal integers
	###
	@staticmethod
	def bi2de(b):
	    n = b.shape[0]
	    outMatrix = np.zeros([n], dtype = int)
	    
	    for k in range(n):
	        outMatrix[k] = sum(1<<i for (i,b) in enumerate(b[k]) if b !=0)
	    
	    return outMatrix

	###
	#   @brief Converts a 1D array of decimal integers to their equivalent binary values
	#
	#   Converts a non-negative decimal integer vector to a matrix. Each row in the 
	#   matrix is the binary form of the corresponding element in the vector. Each
	#   column of the resulting matrix contains a single binary value
	#
	#   @param d 1D array of decimal integers
	#   @param width Number of bits to be used for each decimal integer
	#
	#   @return Matrix where each row represents the decimal value of each element of d
	###
	@staticmethod
	def de2bi(d, width = None):
	    
	    dmx = int(np.amax(d))
	    if width is None:
	        width = max(dmx.bit_length(), 1)
	        
	    outMatrix = np.zeros([len(d), width], dtype = np.int) 
	    for k in range(len(d)):
	        
	        outMatrix[k,:] =  [(d[k] >> i) & 1 for i in range(width)]
	   
	    return outMatrix

	###
	#   @brief Perfoms QAM modulation of the input signal
	#
	#   The alphabet corresponding to the desired QAM modulation is
	#   first obtained to then assign a symbol to each sequence of
	#   bits which are represented by a decimal integer
	#
	#   @param symbolWord Binary sequence represented by a 1D array of integers
	#   @param M Order of the QAM modulation
	#
	#   @return 1D array with modulated signal represented by coordinates
	#   in the complex constellation diagram
	#   @return 1D array representing the QAM alphabet
	#
	###
	@staticmethod
	def qam_modulate(symbolWord, M):

	    order = int(np.sqrt(M)) # number of bits that can be transmitted per symbol
	    quadrat_iq = np.arange(-order + 1, order+1 , step = 2, dtype = np.int) 
	    const = (quadrat_iq[np.newaxis].T + 1j * quadrat_iq).flatten()

	    return const[symbolWord], const

	###
	#   @brief Demodulation of a QAM signal
	#
	#   @param rx_signal 1D array containing complex baseband symbols
	#   @param M Order of the QAM modulation
	#
	#   @return 1D array of decimal integers which represent the demodulated symbols
	###
	@staticmethod
	def qam_demodulate(rx_sig, M):
	    order = int(np.sqrt(M)) # number of bits that can be transmitted per symbol
	    s_real = np.clip(np.around((rx_sig.real + order - 1) / 2), 0, order - 1).astype(np.int)
	    s_imag = np.clip(np.around((rx_sig.imag + order - 1) / 2), 0, order - 1).astype(np.int)   
	    return order*s_real + s_imag 

	###
	#   @brief Perfoms PSK modulation of the input signal
	#
	#   The alphabet corresponding to the desired PSK modulation is
	#   first obtained to then assign a symbol to each sequence of
	#   bits which are represented by a decimal integer
	#
	#   @param symbolWord Binary sequence represented by a 1D array of integers
	#   @param M Order of the PSK modulation
	#
	#   @return 1D array with modulated signal represented by coordinates
	#   in the complex constellation diagram
	#   @return 1D array representing the PSK alphabet
	#
	###
	@staticmethod
	def psk_modulate(symbolWord, M):
	    order = int(np.log2(M))
	    const = np.empty((M), dtype = complex)
	    for m in range(M):
	        I = np.cos(m/M*2*np.pi)
	        Q = np.sin(m/M*2*np.pi)
	        const[m] = I + 1j*Q
	    return const[symbolWord], const

	###
	#   @brief Perfoms PSK demodulation of the received signal
	#
	#   Demodulation is performed using the minimum euclidian distance method,
	#   where the distance between each received symbol and each constellation
	#   symbol is calculated before selecting the one that is closest to zero.
	#   Finally that symbol is converted to its equivalent binary value. 
	#
	#   @param rx_sig 1D array containing complex baseband symbols
	#   @param constellation Alphabet of the corresponding mPSK modulation 
	#
	#   @return 1D binary array
	#
	###
	@staticmethod
	def psk_demodulate(rx_sig, constellation):
	    min_distance_array = abs(rx_sig - constellation[:, None])
	    indexList = min_distance_array.argmin(0)
	    return opticalSignalTransmission.de2bi(indexList) 

	###
	#   @brief Impulse response of a Root Raised Cosine filter
	#   
	#   @param beta Roll-off of the filter
	#   @param span Length of the filter
	#   @param sps Number of samples per symbol
	#
	#   @return 1D array containing the coefficients of the filter
	###
	@staticmethod
	def rrcosine(beta, span, sps):
	    N           =   span*sps 
	    sample_num  =   np.arange(N+1)
	    h_rrc       =   np.zeros(N+1, dtype = float)
	    Ts          =   sps
	    
	    for x in sample_num:
	        t = (x - N/2)
	        if t == 0.0:
	            h_rrc[x] = (1/np.sqrt(Ts))*(1.0 - beta + (4*beta/np.pi))
	        elif (beta != 0) and (t == Ts/(4*beta) or t == -Ts/(4*beta)):
	            h_rrc[x] = (1/np.sqrt(Ts))*(beta/(np.sqrt(2)))*(((1+2/np.pi)*\
	            	(np.sin(np.pi/(4*beta)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*beta)))))
	        else:
	            h_rrc[x] = (1/np.sqrt(Ts))*(np.sin(np.pi*t*(1-beta)/Ts) + 4*beta*(t/Ts)*\
	            	np.cos(np.pi*t*(1+beta)/Ts))/(np.pi*t*(1-(4*beta*t/Ts)*(4*beta*t/Ts))/Ts)

	    return h_rrc

	###
	#   @brief Matched Filtering at the receiver
	#   
	#   @param filter String representing the type of filter at the receiver
	#
	#   @return npol x n matrix representing the filtered signal
	###
	def matchedFilt(self, filter = "rrc"):
		if filter == "rrc":
			signalFilter = self.rrcosine(self.rrcRolloff, self.rrcSpan, self.numSpS)
		'''
		The upfirdn implementation in SciPy follows the same logic as in Matlab's
		making the length of the resulting output of the filter equal to
		[(length(input_signal) - 1) * upsampling_factor + length(filter)] / downsampling_factor
		'''
		rxFiltSignal = np.empty((self.npol, int(np.ceil(((self.shape[1] - 1) + \
			signalFilter.size)/self.numSpS))), dtype = complex)
		for i in range(self.npol):
			rxFiltSignal[i,:] = upfirdn(signalFilter, self[i,:], 1, self.numSpS) # downsampling and matched filtering
		'''
		Because each filtering operation delays the incoming signal by the half the length of the
		filter it is necessary to account for this at reception. Due to the filtering at tx and rx
		this means that the total amount of elements to remove from the received signal is equal 
		to the length of the filter. On the other hand, in order to calculate the BER the length
		of both the tx and rx signals must be the same so again "length of filter" elements are
		removed from the rx signal
		'''
		rxFiltSignal = rxFiltSignal[:, self.rrcSpan:-self.rrcSpan]
		return self.convert_to_signal_class(rxFiltSignal)

	###
	#   @brief Demodulation of the received optical signal
	#
	#	Demodulates the received optical signal (only PSK/QAM modulations supported)
	#
	#	@param None
	#   
	#   @return ndarray filled with the decoded bits
	###
	def demodulation(self):
		dataOut = np.empty((self.npol, self.numBits), int) 
		for i in range(self.npol):
			if 'QAM' in self.mod:
				dataSymbolsOut  = self.qam_demodulate(self[i,:], self.M)
				dataOutMatrix   = self.de2bi(dataSymbolsOut)
			if 'PSK' in self.mod:
				dataOutMatrix = self.psk_demodulate(self[i,:], self.alphabet)
			dataOut[i,:] = dataOutMatrix.flatten()
		return dataOut
