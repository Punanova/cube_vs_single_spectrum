import pyspeckit
from pyspeckit.spectrum.models import n2hp
from spectral_cube import SpectralCube
import numpy as np
from radio_beam import Beam
from astropy.io import fits
import astropy.units as u
from skimage.morphology import remove_small_objects,closing,disk,opening


# Optically thin
Optically_Thin      =False
Show_Optically_Thin =False
# Optically thick
Optically_Thick     =False
Show_Optically_Thick=True

file_raw='data/raw/Core2_n2hp.fits'
file_in='data/Core2_N2Hp_10.fits'
#freq_line=93173.7637e6*u.Hz
freq_line=93173.3920e6*u.Hz

file_thick='fits/Core2_N2Hp_thick_fitted_parameters_snr3.fits'
file_thin= 'fits/Core2_N2Hp_thin_fitted_parameters_snr3.fits'
snr_min = 3.

cube = pyspeckit.Cube(file_in)
cube.xarr.refX = freq_line
cube.xarr.velocity_convention = 'radio'
cube.xarr.convert_to_unit('km/s')

xmax=10; ymax=12
vmin=5.0; vmax=8.0

rms_map = cube.slice(-0.2, 6.6, unit='km/s').cube.std(axis=0)
peaksnr =  cube.slice(vmin, vmax, unit='km/s').cube.max(axis=0)/rms_map
planemask = (peaksnr>snr_min) 
planemask = remove_small_objects(planemask,min_size=40)
planemask = opening(planemask,disk(1))

F=False
T=True
multicore=4

import matplotlib.pyplot as plt
plt.ion()

if Optically_Thin:
    cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)

    print('start optically thin fit')
    cube.fiteach(fittype='n2hp_vtau',  guesses=[5, 0.1, 7.0, 0.1], # Tex=5K, tau=0.1, v_center=7.1, \sigma_v=0.3 km/s
                 verbose_level=1, signal_cut=snr_min,
                 limitedmax=[F,F,T,T],
                 limitedmin=[T,T,T,T],
                 minpars=[ 0,  0,vmin,0.05],
                 maxpars=[35.0,0,vmax,1.0],
                 fixed=[F,T,F,F], 
                 use_neighbor_as_guess=True, 
                 position_order = 1/peaksnr,
                 errmap=rms_map, 
                 multicore=multicore)

    cube.write_fit( file_thin, clobber=True)

if Optically_Thick:
    cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)

    print('start optically thick fit')
    cube.fiteach(fittype='n2hp_vtau',  guesses=[5, 0.07, 6.7, 0.3], # Tex=5K, tau=0.1, v_center=7.1, \sigma_v=0.3 km/s
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


if Show_Optically_Thin:
    #
    cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)
    cube.load_model_fit(file_thin, npars=4, npeaks=1, _temp_fit_loc=(xmax,ymax))
    cube.mapplot()
    cube.plot_spectrum(xmax,ymax, plot_fit=True)
    cube.mapplot.plane = cube.parcube[0,:,:]
    cube.mapplot(estimator=None, vmin=2, vmax=20)
    plt.draw()
    plt.show()


if Show_Optically_Thick:
    #
    cube.Registry.add_fitter('n2hp_vtau', pyspeckit.models.n2hp.n2hp_vtau_fitter, 4)
    cube.load_model_fit( file_thick, npars=4, npeaks=1, _temp_fit_loc=(xmax,ymax))
    cube.mapplot()
    cube.plot_spectrum(xmax,ymax, plot_fit=True)
    cube.mapplot.plane = cube.parcube[0,:,:]
    cube.mapplot(estimator=None, vmin=2, vmax=5)
    plt.draw()
    plt.show()

