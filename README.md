# opticalNet
opticalNet is a tool written in Python (*tested with v3.8.10*) to simulate the transmission and reception of an optical signal. It is written as part of my MSc. Dissertation at the Aston Institute of Photonic Technologies (Aston University, UK). The thesis is called "Modulation Format Identification in Optical Networks using Machine Learning Techniques". Project ongoing as of July 2021
## How to use

* Modulation options: 4-PSK, 8-PSK, 4-QAM, 16-QAM, 64-QAM, 128-QAM, 256-QAM
* Polarization: Single/Dual
* Pulse shaping using RRC filter
* Channel Effects:
	- Rotation of state of polarization
	- Phase noise
	- AWGN
	- Polarization Mode Dispersion
	- Chromatic Dispersion


**USAGE** 
```python
python fileName.py [-m modulation][-sps samples_per_symbol] 
		[-snr signal_to_noise_ratio_rx] [-L length] [-wlength wavelength] 
		[-pmd differential_group_delay] [-cd chromatic_dispersion] -pn -dp
```

**OPTIONS**

- **m** *modulation* Specify desired modulation format: 4-QAM, 16-QAM (default if omitted), 64-QAM or 256-QAM

- **sps** *samples_per_symbol* Number of samples per symbol. Must be an integer. Default = 4

- **snr** *signal_to_noise_ratio_rx* Specify signal to noise ratio (dB) at receiver. Must be an integer. Default = 10 [dB]

- **pn** *phase_noise* Adds phase noise to the signal
          
- **L** *length* Length of the SMF in kilometers. Default = 500 [km]
          
- **wlength** *wavelength* Wavelength of the signal in meters. Default =  1550[nm]

- **dp** *dualpolarization* Transmission of a dually polarized optical signal affected by polarization rotation
				                            which is corrected by means of the CMA algorithm (demux applies only to 4-QAM and mPSK)
                              
- **pmd** *polarization mode dispersion* Differential Group Delay in seconds. It is the difference in propagation time between
                                    the vertical and horizontal polarization modes. Default = 10e-12[s]
                                    
- **cd** *chromatic dispersion* Dispersion induced by the dependence of the index of refraction of the fiber on the
				                            wavelength of the propagating signal. Measured in [s²/m]. Default = 17e-6[s²/m]
