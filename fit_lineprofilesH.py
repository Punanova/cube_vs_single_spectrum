import pyspeckit
from pyspeckit.spectrum.models import n2hp
from spectral_cube import SpectralCube
import numpy as np
from radio_beam import Beam
from astropy.io import fits
import astropy.units as u
from skimage.morphology import remove_small_objects,closing,disk,opening

file_in='data/Core2_N2Hp_10.fits'
freq_line=93173.3922e6*u.Hz

file_thick='fits/Core2_N2Hp_thick_fitted_parameters_snr3.fits'
file_thick_v2='fits/Core2_N2Hp_thick_fitted_parameters_snr3_v2.fits'
snr_min = 3.

cube = pyspeckit.Cube(file_in)
cube.xarr.refX = freq_line
cube.xarr.velocity_convention = 'radio'
cube.xarr.convert_to_unit('km/s')

xmax=9; ymax=9
vmin=5.0; vmax=8.0


# This line is wrong, because it is calculating the rms over the channels with emission
#rms_map = cube.slice(-0.2, 6.6, unit='km/s').cube.std(axis=0) 
rms_map = cube.slice(15.0, 25.0, unit='km/s').cube.std(axis=0) # 
peaksnr =  cube.slice(vmin, vmax, unit='km/s').cube.max(axis=0)/rms_map
planemask = (peaksnr>snr_min) 
planemask = remove_small_objects(planemask,min_size=40)
planemask = opening(planemask,disk(1))

print("calculated rms used at position of maximum is {0} K, and it should be 0.042 K".format(rms_map[ymax,ymax]))

F=False
T=True
multicore=4

import matplotlib.pyplot as plt
plt.ion()

cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)

print('start optically thick fit without error')
cube.fiteach(fittype='n2hp_vtau',  guesses=[5, 0.07, 6.7, 0.3], 
             verbose_level=2, signal_cut=snr_min,
             limitedmax=[F,T,T,T],
             limitedmin=[T,T,T,T],
             minpars=[ 0,  0,vmin,0.05],
             maxpars=[20.,50,vmax,1.0],
             fixed=[F,F,F,F], 
             use_neighbor_as_guess=True, 
             position_order = 1/peaksnr,
             errmap=rms_map, 
             multicore=multicore)
cube.write_fit( file_thick, clobber=True)

print('start optically thick fit with error')
# cube.error[:] = rms_map
cube.fiteach(fittype='n2hp_vtau',  guesses=[5, 0.07, 6.7, 0.3], 
             verbose_level=2, signal_cut=snr_min,
             limitedmax=[F,T,T,T],
             limitedmin=[T,T,T,T],
             minpars=[ 0,  0,vmin,0.05],
             maxpars=[20.,50,vmax,1.0],
             fixed=[F,F,F,F], 
             use_neighbor_as_guess=True, 
             position_order = 1/peaksnr,
             errmap=rms_map*0.5, 
             multicore=multicore)

cube.write_fit( file_thick_v2, clobber=True)

# plot figure
cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)

# No error
cube.load_model_fit( file_thick, npars=4, npeaks=1, _temp_fit_loc=(xmax,ymax))
cube.mapplot()
cube.plot_spectrum(xmax,ymax, plot_fit=True)
cube.mapplot.plane = cube.parcube[0,:,:]
cube.mapplot(estimator=None, vmin=2, vmax=5)
plt.draw()
plt.show()

plt.savefig('N2Hp_pyspeckit_fit_from_cube_with_error.png')

# With error
cube.load_model_fit( file_thick_v2, npars=4, npeaks=1, _temp_fit_loc=(xmax,ymax))
cube.mapplot()
cube.plot_spectrum(xmax,ymax, plot_fit=True)
cube.mapplot.plane = cube.parcube[0,:,:]
cube.mapplot(estimator=None, vmin=2, vmax=5)
plt.draw()
plt.show()

plt.savefig('N2Hp_pyspeckit_fit_from_cube_with_half_error.png')