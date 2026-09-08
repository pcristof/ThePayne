'''
PIC: New Module to read data independently from file name and number 
of dimensions.
The goal is to have a more `general` way of reading spectra, so that we can
have any kind of dimension in the data we provide.
I am building it starting off from readc3k.py, but modifying it substantially'''

# #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os,sys,glob,warnings
import numpy as np
from numpy.lib import recfunctions as rfn
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import h5py
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import beta
from datetime import datetime

import Payne
from .smoothing import smoothspec

class readc3k(object):
    def __init__(self,**kwargs):
        ## modelPath: path to the models
        ## This sould likely be modified in the future
        self.modelPath  = kwargs.get('C3Kpath',Payne.__abspath__+'data/C3K/')
        if self.modelPath is None:
            self.modelPath = Payne.__abspath__+'data/C3K/'
        
        ## This is what I am not a fan of:
        ## the c3k reading function reads the metallicities and other parameters
        ## from the name of the files. I would like the file names to not
        ## matter since the information is already encoded in the hdf5 files.

        ## What I really want to to have a list of labels
        self.PARAMETERS = []
        ## And for each labels I will create an array of values.
        
        ## I am also deleting everything that is related to weights, because
        ## I do not understand what they are doing here (I am assuming these
        ## were to help choose the spectra to pick).

        ## The C3K then created a dictionary that was populated for multiple
        ## metallicities. I prefer going through all the files and dynamically
        ## deciding on the available metallicities.
        ## I understand this may require more time to read; but we could also
        ## simply add the labels and range in the meta-data of the input files
        
        ## Instead, here, I create a dictionary with keys containing all 
        ## parameters
        self.SPECTRA = {}
        filenames = glob.glob(self.modelPath+'*.h5')
        i = 0

        if len(filenames)==0:
            raise Exception('Could not find any model in the input directory.'
                            +f'\nInput provided: {self.modelPath}')

        self.PARAMETERS = None
        ranges = None
        for fname in filenames:
            self.SPECTRA[i] = h5py.File(
                                fname,
                                'r', libver='latest', swmr=True)
            ## Get the labels from the parameters:
            _params = list(self.SPECTRA[i]['parameters'].dtype.names)
            ## From the data, define the boundaries of our grid
            if ranges is None: ## initialize
                ranges = {}
                for _p in _params:
                    _min = np.min(self.SPECTRA[i]['parameters'][_p])
                    _max = np.max(self.SPECTRA[i]['parameters'][_p])
                    ranges[_p] = [_min, _max]
            else: ## Check that we have the whole ranges
                for _p in _params:
                    _min = np.min(self.SPECTRA[i]['parameters'][_p])
                    _max = np.max(self.SPECTRA[i]['parameters'][_p])
                    if _min<ranges[_p][0]: ranges[_p][0] = _min
                    if _max>ranges[_p][1]: ranges[_p][1] = _max
            if self.PARAMETERS is None:
                self.PARAMETERS = _params
            else:
                if _params!=self.PARAMETERS:
                    raise Exception('Mismatch between labels in the '
                                    +'different input files. Please check your '
                                    +'input.')
            i+=1

        self._labels = []
        for file_id in range(len(self.SPECTRA)):
            for local_id in range(len(self.SPECTRA[file_id]['parameters'])):
                _lab = list(self.SPECTRA[file_id]['parameters'][local_id])
                self._labels.append(_lab)
        self._labels = np.array(self._labels)
        
        ## The following is needed for the rest of the program
        ## create min-max dictionary for input labels
        self.minmax = {}
        for _param in self.PARAMETERS:
            ## I am padding the limits that I draw from the parameters
            ## otherwise the program will have problem including the models
            ## at the limits of the grid.
            ## Also note that the limits should NOT be np.inf. Otherwise you
            ## get problems.
            diff_range = ranges[_param][1]-ranges[_param][0]
            pad = diff_range*0.1 ## 1% of the range used for padding
            if pad<0.01: pad=1.0 ## If the values are the same
            self.minmax[_param] = [ranges[_param][0]-pad, ranges[_param][1]+pad]
        
        ## What is this and why are we enforcing 1.0 ???
        # create min-max for spectra
        self.Fminmax = [0.0,1.0]

        self.verbose = kwargs.get('verbose',False)

        # # init random number object
        self.rng = np.random.default_rng()
        self.mean_standard = None

        # Build a lightweight index of the spectra in each HDF5 file.
        # We do NOT load the spectra themselves.

        self._file_ids = []
        self._file_offsets = []

        offset = 0

        for i in range(len(self.SPECTRA)):
            n = self.SPECTRA[i]['parameters'].shape[0]

            self._file_ids.append(i)
            self._file_offsets.append(offset)

            offset += n

        self._file_offsets = np.asarray(self._file_offsets, dtype=np.int64)
        self._nspectra = offset


    def pullspectra(self,num,**kwargs):
        '''
        Randomly draw num spectra from C3K with option to 
        base draw on the MIST isochrones.
        
        :params num:
            Number of spectra randomly drawn 

        :params label (optional):
            kwarg defined as labelname=[min, max]
            This constrains the spectra to only 
            be drawn from a given range of labels

        :params excludelabels (optional):
            kwarg defined as array of labels
            that should not be included in 
            output sample of spectra. Useful 
            for when defining validation and 
            testing spectra

        : params waverange (optional):
            kwarg used to set wavelength range
            of output spectra
        
        : params reclabelsel (optional):
            kwarg boolean that returns arrays that 
            give how the labels were selected

        : params returncontinuua (optional):
            kwarg boolean that returns contiunuua in 
            addition to the normalized spectra

        : returns spectra:
            Structured array: wave, spectra1, spectra2, spectra3, ...
            where spectrai is a flux array for ith spectrum. wave is the
            wavelength array in nm.

        : returns labels:
            Array of labels for the individual drawn spectra

        : returns wavelenths:
            Array of wavelengths for predicted spectra

        '''

        ## I am removing any mention to MIST

        ## The C3K implementation allows to pass ranges for the input parameters
        ## The issue is that if we change those labels and/or parameters, the
        ## we would have to change this implementation.
        ## Instead, I propose to implement the possibility of passing two lists:
        ## one containing the labels of the parameters whose range we want to
        ## constrain, and another containing the limits of that constraint.
        ## Default would be no constraint.

        labelsToConstrain = kwargs.get('labelsToConstrain',None)
        paramRanges = kwargs.get('paramRanges',None)

        if labelsToConstrain is not None:
            for l in labelsToConstrain:
                if l not in self.PARAMETERS:
                    raise Exception('The labels you want to constrain are not '
                                    +'in the input files. The labels that we have '
                                    +'are:\n' + self.PARAMETERS)

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'excludelabels' in kwargs:
            excludelabels = kwargs['excludelabels'].T.tolist()
        else:
            excludelabels = []


        # default is just the MgB triplet 
        waverange = kwargs.get('waverange',[5150.0,5300.0])

        # set up some booleans
        dividecont    = kwargs.get('dividecont',True)
        reclabelsel   = kwargs.get('reclabelsel',False)
        continuuabool = kwargs.get('returncontinuua',False)
        spectrumMode = kwargs.get('spectrumMode','spectra')
        # timeit        = kwargs.get('timeit',False)
        

        ## ONE ISSUE is the time it takes to read in the models.
        ## We should be able to do faster by not creating huge
        ## lists all the time.
        # from IPython import embed;embed()

        ## Dummy variables for now.
        self.mean_standard = 0.0
        self.std_standard = 1.0
        self.normFactor = 1.0

        available_indices = list(np.arange(self._nspectra, dtype=int))
        ## Get a list of allowed indices
        allowed = np.ones(self._nspectra, dtype=bool)
        excludelabels_T = [list(row) for row in zip(*excludelabels)]
        for label in excludelabels_T:
            allowed &= ~np.all(
                self._labels == label,
                axis=1
            )
        allowed_indices = np.flatnonzero(allowed)

        indices = self.rng.choice(
            allowed_indices,
            size=num,
            replace=False
            )

        ## Now I need to decode this index in terms of file number and position
        ## Now identify the file in which this index is:
        file_id = np.searchsorted(self._file_offsets,
                                    indices, side='right' ) - 1
        ## and the local id inside that file
        local_id = (indices - self._file_offsets[file_id])

        labels=[]
        spectra=[]
        for i in range(len(file_id)):
            _lab = self._labels[indices[i]]
            spectra_i = self.SPECTRA[file_id[i]][spectrumMode][local_id[i]]

            wavecond = np.ones(len(spectra_i), dtype=bool)
            wavelength_o = np.arange(len(wavecond))

            # if user defined resolution to train at, the smooth C3K to that resolution
            if resolution != None:
                spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
                    outwave=wavelength_o,smoothtype='R',fftsmooth=True)
            else:
                spectra_i = spectra_i[wavecond]

            if continuuabool:
                if resolution != None:
                    continuua_i = self.smoothspecfunc(wavelength_i,continuua_i,resolution,
                        outwave=wavelength_o,smoothtype='R',fftsmooth=True)
                else:
                    continuua_i = continuua_i[wavecond]

            labels.append(_lab)
            spectra.append(spectra_i)
            # if requested, return continuua
            if continuuabool:
                continuua.append(continuua_i)

            if self.verbose:
                print(f'-> Added {i+1}, total time: {0}'.format(datetime.now()-starttime))
            ## Increase iterator
            # ii+=1
        output = [np.array(spectra), np.array(labels),wavelength_o]

        # if reclabelsel:
        #     output += [np.array(initlabels)]
        # if continuuabool:
        #     output += [np.array(continuua)]

        return output

    def selspectra(self,inlabels,**kwargs):
        '''
        specifically select and return C3K spectra at user
        defined labels

        :param inlabels
        Array of user defined lables for returned C3K spectra
        format is [Teff,logg,FeH,aFe]

        '''

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'waverange' in kwargs:
            waverange = kwargs['waverange']
        else:
            # default is just the MgB triplet 
            waverange = [5150.0,5200.0]

        # user wants to return continuua
        continuuabool = kwargs.get('returncontinuua',False)
        dividecont    = kwargs.get('dividecont',True)

        labels = []
        spectra = []
        wavelength_o_flag = True

        if continuuabool:
            continuua = []

        if isinstance(inlabels[0],float):
            inlabels = [inlabels]

        for li in inlabels:
            # select the C3K spectra at that [Fe/H] and [alpha/Fe]
            teff_i  = li[0]
            logg_i  = li[1]
            FeH_i   = li[2]
            alpha_i = li[3]

            if len(self.vtarr) > 0:
                vt_i = li[4]

            # find nearest value to FeH and aFe
            try:
                FeH_i   = self.FeHarr[np.argmin(np.abs(np.array(self.FeHarr)-FeH_i))]
            except:
                print('Issue with finding nearest FeH')
                print(self.FeHarr)
                print(FeH_i)
                raise

            try:
                alpha_i = self.alphaarr[np.argmin(np.abs(np.array(self.alphaarr)-alpha_i))]
            except:
                print('Issue with finding nearest aFe')
                print(self.alphaarr)
                print(alpha_i)
                raise				

            if len(self.vtarr) > 0:
                # select the C3K spectra at that [Fe/H], [alpha/Fe], vturb
                C3K_i = self.C3K[alpha_i][FeH_i][vt_i]
                # create array of all labels in specific C3K file
                C3Kpars = np.array(C3K_i['parameters'])
                # tack on vturb to parameter array
                C3Kpars = rfn.rec_append_fields(
                    C3Kpars,'vt',
                    vt_i*np.ones(C3Kpars.shape[0],dtype=float),
                    dtypes=float)
            else:
                # select the C3K spectra at that [Fe/H] and [alpha/Fe]
                C3K_i = self.C3K[alpha_i][FeH_i]
                # create array of all labels in specific C3K file
                C3Kpars = np.array(C3K_i['parameters'])

            # convert log(teff) to teff
            C3Kpars['logt'] = 10.0**C3Kpars['logt']
            C3Kpars = rfn.rename_fields(C3Kpars,{'logt':'teff'})

            # do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
            C3KNN = NearestNDInterpolator(
                np.array([C3Kpars['teff'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
                )((teff_i,logg_i))
            C3KNN = int(C3KNN)

            # determine the labels for the selected C3K spectrum
            try:
                label_i = list(C3Kpars[C3KNN])
            except IndexError:
                print(C3KNN)
                raise

            # turn off warnings for this step, C3K has some continuaa with flux = 0
            if dividecont:
                with np.errstate(divide='ignore', invalid='ignore'):
                    spectra_i = C3K_i['spectra'][C3KNN]/C3K_i['continuua'][C3KNN]
            else:
                spectra_i = C3K_i['spectra'][C3KNN]/np.nanmedian(C3K_i['spectra'][C3KNN])
                
            if continuuabool:
                continuua_i = C3K_i['continuua'][C3KNN]
            else:
                continuua_i = None

            # check to see if label_i in labels, or spectra_i is nan's
            # if so, then skip the append and go to next step in while loop
            # do this before the smoothing to reduce run time
            # if np.any(np.isnan(spectra_i)):
            # 	continue

            # store a wavelength array as an instance, all of C3K has 
            # the same wavelength sampling
            if wavelength_o_flag:
                wavelength_o = [] # initialize the output wavelength array
                wavelength_o_flag = False # turn off this step for all subsequent models
                wavelength_i = np.array(C3K_i['wavelengths'])
                if resolution != None:
                    # define new wavelength array with 3*resolution element sampling
                    i = 1
                    while True:
                        wave_i = waverange[0]*(1.0 + 1.0/(3.0*resolution))**(i-1.0)
                        if wave_i <= waverange[1]:
                            wavelength_o.append(wave_i)
                            i += 1
                        else:
                            break
                    wavelength_o = np.array(wavelength_o)
                else:
                    wavecond = (wavelength_i >= waverange[0]) & (wavelength_i <= waverange[1])
                    wavecond = np.array(wavecond,dtype=bool)
                    wavelength_o = wavelength_i[wavecond]

            # if user defined resolution to train at, the smooth C3K to that resolution
            if resolution != None:
                spectra_i = self.smoothspecfunc(wavelength_i,spectra_i,resolution,
                    outwave=wavelength_o,smoothtype='R',fftsmooth=True)
            else:
                spectra_i = spectra_i[wavecond]

            if continuuabool:
                if resolution != None:
                    continuua_i = self.smoothspecfunc(wavelength_i,continuua_i,resolution,
                        outwave=wavelength_o,smoothtype='R',fftsmooth=True)
                else:
                    continuua_i = continuua_i[wavecond]

            labels.append(label_i)
            spectra.append(spectra_i)
            if continuuabool:
                continuua.append(continuua_i)

        output = [np.array(spectra), np.array(labels), wavelength_o]

        if continuuabool:
            output += [np.array(continuua)]

        return output

    def pullpixel(self,pixelnum,**kwargs):
        # a convience function if you only want to pull one pixel at a time
        # it also does a check and remove any NaNs in spectra 


        Teffrange = kwargs.get('Teff',None)
        if Teffrange == None:
            Teffrange = [2500.0,15000.0]

        loggrange = kwargs.get('logg',None)
        if loggrange == None:
            loggrange = [-1.0,5.0]

        fehrange = kwargs.get('FeH',None)
        if fehrange == None:
            fehrange = [min(self.FeHarr),max(self.FeHarr)]

        aFerange = kwargs.get('aFe',None)
        if aFerange == None:
            aFerange = [min(self.alphaarr),max(self.alphaarr)]

        if 'resolution' in kwargs:
            resolution = kwargs['resolution']
        else:
            resolution = None

        if 'excludelabels' in kwargs:
            excludelabels = kwargs['excludelabels']
        else:
            excludelabels = []

        if 'waverange' in kwargs:
            waverange = kwargs['waverange']
        else:
            # default is just the MgB triplet 
            waverange = [5145.0,5300.0]

        if 'reclabelsel' in kwargs:
            reclabelsel = kwargs['reclabelsel']
        else:
            reclabelsel = False

        if 'MISTweighting' in kwargs:
            MISTweighting = kwargs['MISTweighting']
        else:
            MISTweighting = True

        # if 'timeit' in kwargs:
        # 	timeit = kwargs['timeit']
        # else:
        # 	timeit = False

        if 'inlabels' in kwargs:
            inlabels = kwargs['inlabels']
        else:
            inlabels = []

        if 'num' in kwargs:
            num = kwargs['num']
        else:
            num = 1

        if inlabels == []:				
            # pull the spectrum
            spectra,labels,wavelength = self(
                num,resolution=resolution, waverange=waverange,
                MISTweighting=MISTweighting)

        else:
            spectra,labels,wavelength = self.selspectra(
                inlabels,resolution=resolution, waverange=waverange)

        # select individual pixels
        pixelarr = np.array(spectra[:,pixelnum])
        labels = np.array(labels)

        # # determine if an of the pixels are NaNs
        # mask = np.ones_like(pixelarr,dtype=bool)
        # nanval = np.nonzero(np.isnan(pixelarr))
        # numnan = len(nanval[0])
        # mask[np.nonzero(np.isnan(pixelarr))] = False

        # # remove nan pixel values and labels
        # pixelarr = pixelarr[mask]
        # labels = labels[mask]
        
        return pixelarr, labels, wavelength

    def checklabels(self,inlabels,**kwargs):
        # a function that allows the user to determine the nearest C3K labels to array on input labels
        # useful to run before actually selecting spectra

        labels = []

        for li in inlabels:
            # select the C3K spectra at that [Fe/H] and [alpha/Fe]
            teff_i  = li[0]
            logg_i  = li[1]
            FeH_i   = li[2]
            alpha_i = li[3]

            # find nearest value to FeH and aFe
            FeH_i   = self.FeHarr[np.argmin(np.abs(self.FeHarr-FeH_i))]
            alpha_i = self.alphaarr[np.argmin(np.abs(self.alphaarr-alpha_i))]

            # select the C3K spectra for these alpha and FeH
            C3K_i = self.C3K[alpha_i][FeH_i]

            # create array of all labels in specific C3K file
            C3Kpars = np.array(C3K_i['parameters'])

            # do a nearest neighbor interpolation on Teff and log(g) in the C3K grid
            C3KNN = NearestNDInterpolator(
                np.array([C3Kpars['logt'],C3Kpars['logg']]).T,range(0,len(C3Kpars))
                )((teff_i,logg_i))
            C3KNN = int(C3KNN)

            # determine the labels for the selected C3K spectrum
            label_i = list(C3Kpars[C3KNN])		
            labels.append(label_i)

        return np.array(labels)


    def smoothspecfunc(self,wave, spec, sigma, outwave=None, **kwargs):
        outspec = smoothspec(wave, spec, sigma, outwave=outwave, **kwargs)
        ## I find this really dumb, but this function is not simply bypassed
        ## Will potentially revise this in the future
        # from IPython import embed;embed()
        return outspec
