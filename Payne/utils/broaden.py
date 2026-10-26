import numpy as np
from numba import jit ## OPTIONAL - numba for extremely fast computations

def __lsf_rotate(deltav,vsini,epsilon=0.6):
    '''Function to compute a rotation kernel to convolve a spectrum with.
    Tested against proper disk integration and yields excellent performances.
    '''
    # Based on lsf_rotate.pro:
    #  http://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro
    #
    # Adapted from rotin3.f in the SYNSPEC software of Hubeny & Lanz
    # http://nova.astro.umd.edu/index.html    Also see Eq. 17.12 in
    # "The Observation and Analysis of Stellar Photospheres" by D. Gray (1992)
    ## The equation is rather 18.14
    ## Correct

    ## The number of points that we need
    npts = np.floor(2*vsini/deltav)
    if npts % 2 == 0:
        npts += 1
    nwid = np.floor(npts/2)
    # nwid = npts/2
    x = np.arange(npts) - nwid
    ## Transform this axis in units of deltav (here km/s)
    x = x*deltav
    ## The maximum velocity is vsini
    maxvel = vsini
    ## Compute the different part fo the equation
    e1 = 2.0*(1.0 - epsilon)
    ## Correct
    e2 = np.pi*epsilon/2.0
    ## Not correct? Must add maximum shift !
    e3 = np.pi*(1.0 - epsilon/3.0)*maxvel
    # ## Should be?
    # e3 = 1.0/np.pi/deltav/(1.0-epsilon)/3.
    ##
    x1 = np.abs(1.0 - (x/vsini)**2)

    velgrid = x#*maxvel
    ## wrong?
    ker = (e1*np.sqrt(x1) + e2*x1)/e3
    ## Should be?
    # ker = (e1*np.sqrt(x1) + e2*x1)*e3

    return velgrid, ker

@jit(nopython=True, cache=True)
def _rt_macro(velocity_step, vmac, m):
    """
    Function to compute the RT kernel locally
        velocity_step: fluxes should correspond to a spectrum homogeneously sampled in velocity space
                    with a fixed velocity step [km/s]
        vmac   : macroturbulence velocity [km/s]

        Based on SME's rtint
        It uses radial-tangential instead of isotropic Gaussian macroturbulence.
    """
    # if vmac is not None and vmac > 0:
    # mu represent angles that divide the star into equal area annuli,
    # ordered from disk center (mu=1) to the limb (mu=0).
    # But since we don't have intensity profiles at various viewing (mu) angles
    # at this point, we just take a middle point:
#         m = 0.5
    # Calc projected simga for radial and tangential velocity distributions.
    sigma = vmac/np.sqrt(2.0) / velocity_step
    sigr = sigma * m
    sigt = sigma * np.sqrt(1.0 - m**2.)
    # Figure out how many points to use in macroturbulence kernel
    # nmk = max(min(round(sigma*10), (len(flux)-3)/2), 3)
    nmk = max(round(sigma*10), 3)
    # nmk = round(sigma*10)
    # Construct radial macroturbulence kernel w/ sigma of mu*vmac/sqrt(2)
    if sigr > 0:
        xarg = (np.arange(2*nmk+1)-nmk) / sigr   # exponential arg
        #mrkern = np.exp(max((-0.5*(xarg**2)),-20.0))
        mrkern = np.exp(-0.5*(xarg**2))
        mrkern = mrkern/mrkern.sum()
    else:
        mrkern = np.zeros(2*nmk+1)
        mrkern[nmk] = 1.0    #delta function

    # Construct tangential kernel w/ sigma of sqrt(1-mu**2)*vmac/sqrt(2.)
    if sigt > 0:
        xarg = (np.arange(2*nmk+1)-nmk) /sigt
        mtkern = np.exp(-0.5*(xarg**2))
        mtkern = mtkern/mtkern.sum()
    else:
        mtkern = np.zeros(2*nmk+1)
        mtkern[nmk] = 1.0

    ## Sum the radial and tangential components, weighted by surface area
    area_r = 0.5
    area_t = 0.5
    mkern = area_r*mrkern + area_t*mtkern

    return mkern

@jit(nopython=True, cache=True)
def rt_macro(velocity_step, vmac):
    '''Function to compute a disk-integrated macroturbulence kernel.
    CAUTION: This one DOES not account for limb darkening.'''
    itot = 9
    dskdj = 0
    count = 0
    kernel = _rt_macro(velocity_step, vmac, 0)
    for dski in range(1,10):
        ju = 6*dski
        dskri = float(dski)
        ritot = float(itot)
        dskr = (dskri - 0.5)/ritot
        area = np.pi*(2.0*dskri - 1.0)/(6.0*dskri*(ritot**2))
        subkern = _rt_macro(velocity_step, vmac, abs(dskr))
        ## And here instead of just looping, we put a factor
        subkern = ju * subkern/area
        kernel = kernel + subkern
        count = count + ju
    kernel = kernel / count
    return kernel

@jit(nopython=True)
def gaussian(sigma):
    x = np.arange(16*sigma+1)-8*sigma
    x = np.arange(17*sigma)#-8*sigma
    if x.size%2==0:
        x = x[:-1]
    x = x - len(x)//2
    #     x = np.arange(20*sigma+sigma)-11*sigma
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x)**2/(2*sigma**2))

def broaden_spectrum(wvl, flux, vinstru=0, vsini=0, epsilon=0.6, 
                     vmac=0):
    '''This is the clean function to use to broaden spectra.
    This function was adapted from ASAP v0.1 [https://github.com/pcristof/asap]
    This function broadens the input spectrum to account for a Gaussian instrumental width, rotation, macroturbulence.
    Input parameters:
    - wvl       : [1D array] wavelength solution
    - flux      : [1D array] spectrum to broaden
    - vinstru   : [float]    instrumental width (FWHM) assuming Gaussian like shape [km/s]
    - vsini     : [float]    projected rotational velocity [km/s]
    - epsilon   : [float]    limb darkening coefficient for rotational broadening
    - vmac      : [float]    Radial-tangential macroturbulence velocity [km/s]
    NOTES ON IMPLEMENTATION:
    - Macroturbulence can be tricky and should formally require disk integration. I found that
      disk integration of the kernel then convolution with the spectrum provides good agreement with
      formal disk integration.
    Last update Sep. 15, 2022 in ASAP
    Extracted on Oct. 6, 2026
    '''

    speed_of_light = 299792458.0

    oflux = flux.copy()
    vmacRT = vmac

    ## Define the wavelength (array assumed constant in velocity)
    ## vel_step is computed at the middle of the wavelength array.
    vel_step = (wvl[len(wvl)//2]-wvl[len(wvl)//2-1]) / \
        ((wvl[len(wvl)//2]+wvl[len(wvl)//2-1])/2) * speed_of_light*1e-3 ## In Km/s

    ## Convert vmac and vsini to standard deviation
    vgauss = vinstru/(2*np.sqrt(2*np.log(2))) # FWHM to standard deviation
    ## vgaus in pixel length units:
    sigma_gauss = vgauss/vel_step

    ## Compute the gaussian kernel
    if sigma_gauss <= 0:
        gauss_kernel = None
        len_gauss_kernel = 0
    elif sigma_gauss > 0:
        gauss_kernel = gaussian(sigma_gauss)
        len_gauss_kernel = len(gauss_kernel)
        gauss_kernel /= gauss_kernel.sum()

    ## Compute the roation kernel
    if vsini==0:
        rot_kernel = None
        len_rot_kernel = 0
    else:
        if vsini<0: vsini = -vsini
        _, rot_kernel = __lsf_rotate(vel_step, vsini, epsilon=epsilon)
        len_rot_kernel = len(rot_kernel)
        ## Normalize the kernel
        area = rot_kernel.sum()
        rot_kernel /= area

    ## Compute the RT kernel
    if vmacRT>0:
        rtkernel = rt_macro(vel_step, vmacRT)
        len_rtkernel = len(rtkernel)
        rtkernel /= rtkernel.sum()
    else:
        rtkernel = None
        len_rtkernel = 0

    ## Perform convolutions
    ## For speed, I first compute a masterkernel
    masterkernel = np.array([1]) ## We start with a DIRAC
    if gauss_kernel is not None:
        masterkernel = np.convolve(masterkernel, gauss_kernel, mode='full')
    if rot_kernel is not None:
        masterkernel = np.convolve(masterkernel, rot_kernel, mode='full')
    if rtkernel is not None:
        masterkernel = np.convolve(masterkernel, rtkernel, mode='full')
    ## To prevent errors, if the spectrum is shorter than the kernel,
    ## I pad the input spectrum with ones (assuming normalized spectrum).
    ## One could alternatively raise an error if that is the case.
    if len(oflux)<len(masterkernel):
        diff = len(masterkernel)-len(oflux)
        edge = 2*diff
        oflux_tmp = np.pad(oflux, edge, constant_values=1) 
        oflux_tmp = np.convolve(oflux_tmp, masterkernel, mode='same')
        oflux = oflux_tmp[edge:-edge]
    else:
        oflux = np.convolve(oflux, masterkernel, mode='same')

    return oflux
