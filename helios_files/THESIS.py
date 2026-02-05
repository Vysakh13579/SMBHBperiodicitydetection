import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("Python").setLevel(logging.CRITICAL)
logging.getLogger("Coding").setLevel(logging.CRITICAL)
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
import matplotlib.transforms as transforms

from tinygp import kernels
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from prettytable import PrettyTable

from stingray.simulator import simulator
import random
import jaxns
from scipy.signal import savgol_filter

#from jaxns import ExactNestedSampler
from jaxns import DefaultNestedSampler
from jaxns import TerminationCondition

import tensorflow_probability.substrates.jax.distributions as tfpd

from tinygp import GaussianProcess
from functools import partial
import warnings
warnings.filterwarnings('ignore')
import time
import jaxlib




class ModelComparison:
    def __init__(self, time, flux, flux_err, verbose=True):
        '''
        Creates a class object to compare JAXNS models

        Parameters:
        ----------------------------------
        time        :   time data of the lightcurve
        flux        :   Flux data of the lightcurve
        flux_err    :   Flux error data of the lightcurve
        
        '''
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.nsmodels = {}
        self.nssamplers = {}
        self.nsresults = {}
        self.verbose = verbose
        if self.verbose:
            print('Model Comparion object created.')
    
    def add_NestedModel(self, nsmodel, name):
        '''
        Function to add a Nested sampling JAXNS model and provide it a name
        '''
        if isinstance(nsmodel, jaxns.model.Model):
            if name in list(self.nsmodels.keys()):
                print('Model name already exists.')
            else:
                self.nsmodels[name] = nsmodel
                if self.verbose:
                    print('JAXNS model added.')
        else:
            print('Stop being stupid and add a JAXNS model than a model of', type(nsmodel))
        
    def delete_NestedModel(self, name):
        '''
        Function to delete a given model by its name
        '''
        del self.nsmodels[name]
    
    def models(self, print_params=False):
        '''
        Function to list all the models created
        
        '''
        if print_params:
            print(list(self.nsmodels.keys()))
        return list(self.nsmodels.keys()) 

    def pretty_print(self):
        '''
        A function to print all the caculated results
        '''
        
        if len(self.nsmodels.keys())==0:
            print('No models added.')
            return 0

        else:
            #print(self.nsmodels.keys())
            x = PrettyTable()
            x.field_names = np.concatenate(([' '], list(self.nsmodels.keys())))
            par_vals = ['Number of Parameters']
            for key, value in self.nsmodels.items(): 
                par_vals.append(value.U_ndims)
            x.add_row(par_vals)
            
            termination_conditions = np.array(['Reached max samples',
                                                    'Evidence uncertainty low enough',
                                                    'Small remaining evidence',
                                                    'Reached ESS',
                                                    "Used max num likelihood evaluations",
                                                    'Likelihood contour reached',
                                                    'Sampler efficiency too low',
                                                    'All live-points are on a single plateau (potential numerical errors, consider 64-bit)'])

            if len(self.nsresults.keys())>0:
                par_names = ['Termination condition', 'logZ', 'H', 'ESS', ]
            
                row_list = ['Termination condition']
                for key, value in self.nsresults.items():
                    row_list.append(termination_conditions[np.bool_(self._bit_mask(value.termination_reason))])
                x.add_row(row_list)

                row_list = ['logZ']
                for key, value in self.nsresults.items():
                    row_list.append(value.log_Z_mean)
                x.add_row(row_list)

                row_list = ['H']
                for key, value in self.nsresults.items():
                    row_list.append(value.H_mean)
                x.add_row(row_list)

                row_list = ['ESS']
                for key, value in self.nsresults.items():
                    row_list.append(value.ESS)
                x.add_row(row_list)
            print(x)

            print('\n\nModel comparison results')
            if len(self.nsresults.keys())>1:
                bayes, _= self.modelCOMP()
                x = PrettyTable()
                x.field_names = np.concatenate(([' '], list(bayes.keys())))
                row_list = ['Z1/Z2']
                for key, value in bayes.items():
                    row_list.append('{:0.3e}'.format(value))
                x.add_row(row_list)

                row_list = ['log10(Z1/Z2)']
                for key, value in bayes.items():
                    row_list.append(np.round(np.log10(value), decimals=4))
                x.add_row(row_list)

                print(x)

            jax.clear_caches() #To stop the memory build-up


    def _bit_mask(self, int_mask, width=8):
        '''
        Copied from JAXNS sourcecode

        Convert an integer mask into a bit-mask. I.e. convert an integer into list of left-starting bits.

        Examples:

        1 -> [1,0,0,0,0,0,0,0]
        2 -> [0,1,0,0,0,0,0,0]
        3 -> [1,1,0,0,0,0,0,0]

        Args:
            int_mask: int
            width: number of output bits

        Returns:
            List of bits from left
        '''
        return list(map(int, '{:0{size}b}'.format(int_mask, size=width)))[::-1]
    
    def _print_termination_condition(self, val):
        '''
        Function to convert the termination_reason variable to condition. Based on JAXNS source code.
        '''
        termination_bit_mask = self._bit_mask(val)
        for bit, condition in zip(termination_bit_mask, ['Reached max samples',
                                                        'Evidence uncertainty low enough',
                                                        'Small remaining evidence',
                                                        'Reached ESS',
                                                        "Used max num likelihood evaluations",
                                                        'Likelihood contour reached',
                                                        'Sampler efficiency too low',
                                                        'All live-points are on a single plateau (potential numerical errors, consider 64-bit)']):
            if bit==1:
                print('TerminationCondition : ', condition)
        print('\n')
    
    def run_NSsampler(self,name=None, num_par_samplers=1):
        '''
        Function to create and run the nested sampler for each models created.
        Parameters:
        ----------------------------
        name(optional)     : if provided, instead of sampling all the models created only the specified model will be sampled.
                                By default, all the models created are sampled.
        '''
        if name==None:
            if self.verbose:
                print('Total number of models being sampled : ', len(self.nsmodels.keys()))
                print('--------------------------------------------------------------------------')
            for key, value in self.nsmodels.items():
                if self.verbose:
                    print('Model being sampled : ' ,key)
                    print('-------------------------------------------------------------------------')
                
                #  Create the Nested sampler
                #self.nssamplers[key] = DefaultNestedSampler(model=value, num_live_points=200, max_samples=1e4, num_parallel_workers= num_par_samplers)
                self.nssamplers[key] = DefaultNestedSampler(model = value, max_samples = 1e4, num_live_points = 400,  num_parallel_workers= num_par_samplers)
                
                if self.verbose:
                    print('DefaultNestedSampler Created. \nSampling inititated.')
                
                # Run the sampler
                termination_reason, state =self.nssamplers[key](jax.random.PRNGKey(42),
                                        term_cond=TerminationCondition(live_evidence_frac=1e-4))
            
                # Store the results
                self.nsresults[key] = self.nssamplers[key].to_results(state =state, termination_reason =termination_reason)
                if self.verbose:
                    print('Sampling finished.')
                    self._print_termination_condition(self.nsresults[key].termination_reason)

                jax.clear_caches()

        else:
            # Same as above but since 'name' is provided only running it for one model
            if self.verbose:
                    print('Model being sampled : ' ,name)
                    print('-------------------------------------------------------------------------')
            #self.nssamplers[name] = DefaultNestedSampler(model = self.nsmodels[name], num_live_points=200, max_samples=1e4, num_parallel_workers= num_par_samplers)
            self.nssamplers[name] = DefaultNestedSampler(model = self.nsmodels[name], max_samples = 1e4,  num_live_points = 400,num_parallel_workers= num_par_samplers)
            
            if self.verbose:
                    print('DefaultNestedSampler Created. \nSampling inititated.')
            termination_reason, state =self.nssamplers[name](jax.random.PRNGKey(42),
                                        term_cond=TerminationCondition(live_evidence_frac=1e-4))
            self.nsresults[name] = self.nssamplers[name].to_results(state = state, termination_reason = termination_reason)
            if self.verbose:
                    print('Sampling finished.')
                    self._print_termination_condition(self.nsresults[name].termination_reason)
                
            jax.clear_caches()
            
    def _convert_numpy_arrays_to_lists(self, input_dict):
        output_dict = {}
        
        for key, value in input_dict.items():
            if isinstance(value, jaxlib.xla_extension.ArrayImpl):
                output_dict[key] = value.tolist()
    
            else:
                output_dict[key] = value
        
        return output_dict
    
    def return_samples_logZ_dict(self, model_name):
        '''
        Function to return all the samples of the parameters and the logZ value of the given model as a dict
        '''
        
        model_results = self.nsresults[model_name]
        
        result_dict= {}
        result_dict['log_Z_mean'] = model_results.log_Z_mean
        result_dict['log_Z_uncert'] = model_results.log_Z_uncert
        result_dict.update(model_results.samples)
        
        result_dict = self._convert_numpy_arrays_to_lists(result_dict)
        
        return result_dict
        
    def modelCOMP(self):
        '''
        A function to calculate and return the Bayes factor in all combinations of models and return the Bayes factor and log Z
        '''
        if len(self.nsresults.keys())<=1:
            print('Add more than 1 model to perform comparion')
        else:
            logZ = {}
            for key, value in self.nsresults.items():
                logZ[key] = value.log_Z_mean
            
            logZ = dict(sorted(logZ.items(), key=lambda item: item[1]))

            BayesFactor = {}
            
            for i in range(len(logZ.values())):
                for j, (keys, value) in enumerate(logZ.items()):
                    if j>i:
                        BayesFactor[keys+'/'+list(logZ.keys())[i]] = np.exp(value - list(logZ.values())[i])
            
            jax.clear_caches()
            return BayesFactor, logZ
        
        

    def print_summary(self, name):
        '''
        function to print the summary of the sampling for a given model
        '''
        print(self.nssamplers[name].summary(self.nsresults[name]))

    def return_sampler(self, name):
        '''
        Function to return the sampler for a given name
        '''
        return self.nssamplers[name]
    
    def return_results(self, name):
        '''
        Function to return the results of sampling for a given name
        '''
        return self.nsresults[name]
    
    
    
    






















class LightCurveSampler:
    def __init__(self, N =2**10, mean = 1.0, dt=10.0, rms=1.0, simulatorSEED = 1079, verbose = True):
        '''
        Creates a class object to generate and sample the light curve using user-defined power spectrum and a user-defined observation window 

        Calling this automatically creates a stingray simulator object with the given set of parameters for further user defined simulation.
        
        Parameters:
        -------------------
        N           :     Number of time points: default value = 10
        mean        :     mean value of lightcurve: default value = 0.5
        dt          :     Time resolution of the lightcurve : default value = 10 minutes
        rms         :     rms value of the lightcurve : default value = 1
        simRANDOMkey   :     Set the random seed for just the simulation part
        verbose     :     toggle verbose
        
        '''
        self.simSeed = simulatorSEED
        self.N = N                     # Number of time points, defaul
        self.mean = mean               # mean value of lightcurve
        self.mins2days = 1/(60.0*24.0) # minutes to days convertion factor
        self.months2days = 30          # months to days converstion factor
        self.OneMonthIndex   = 30/(dt/60/24)        # 30/(10/60/24), i.e., index corresponding to one month interval in the lc sim
        self.dt = dt                   # time resolution 
        self.rms = rms                 # rms value 
        self.verbose = verbose
        self.lc = 0                    # Variable for storing stingray lightcurve

        

        if self.verbose:
            np.random.seed(self.simSeed)
            self.sim = simulator.Simulator(N=self.N, mean=self.mean, dt=self.dt*self.mins2days, rms=self.rms)
    
            self.freq = np.fft.rfftfreq(self.sim.N, d=self.sim.dt)[1:]   # [1:] to remove the DC power in the spectrum

            print('Model parameters')
            print('----------------------------------')
            
            print('Number of time points \t\t\t: ', self.N)
            print('Mean value of lightcurve \t\t: ', self.mean)
            print('Time resolution (mins) \t\t\t: ', self.dt)
            print('RMS value of the Lightcurve\t\t: ', self.rms)

            print('\nStingray simulator object created')
            print('----------------------------------')
            print('Min frequency : ', np.min(self.freq), ' day^-1s')
            print('Max frequency : ', np.max(self.freq), ' day^-1')
            print('lightcurve time coverage : ', self.N*self.dt*self.mins2days/365,' years')

    
    def __call__(self):
        print('----------------------------------------------------------------------------------------------------')
        print('Let\'s makes some lightcurves!!\nThis object lets you generate and sample a light curve using\nuser-defined power spectrum and a user defined observation window')
        print('----------------------------------------------------------------------------------------------------\n\n')
        
        print('Model parameters')
        print('----------------------------------')
        
        print('Number of time points \t\t\t: ', self.N)
        print('Mean value of lightcurve \t\t: ', self.mean)
        print('Time resolution (mins) \t\t\t: ', self.dt)
        print('RMS value of the Lightcurve\t\t: ', self.rms)
        

    def plot_vertical_line(self, ax, x_position, text, c='red', ypos = 0.2, boxalpha = 1, textalpha=0.3):
        ax.axvline(x=x_position, color=c, linestyle='--', alpha=0.3, zorder=1)
        
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        
        ax.text(x_position, ypos, text, color=c, bbox = dict(facecolor = 'white',edgecolor = 'none', alpha =boxalpha),
                ha='center', va='center', transform=trans, rotation = 90, alpha=textalpha, zorder=2, weight='bold')
        
    def data_to_axes_coordinates(self, ax, x, y):
        trans_data_to_axes = ax.transAxes.inverted()
        x_axes, y_axes = trans_data_to_axes.transform((x, y))
        return x_axes, y_axes

    def load_powerspec(self, powmodel, modelparams, plot=False):
        '''
        Function to load the power spectrum model to generate the lightcurve.
        
        Parameters:
        ----------------------

        powmodel        : a funtion, which is of the form 'fun(freq, *params)', which returns the power spectrum
                            freq here should be the frequency and the params be the necessary parameters of the model
        modelparams     : parameters for the powmodel as an array
        plot (optional) : boolean variable to whether display the plot of the power spectrum or not.
                    
        '''
        np.random.seed(self.simSeed)
        self.powmodel = powmodel
        self.powmodelparams = modelparams

        if self.verbose:
            print('Power spectrum model loaded.')

        if plot:
            np.random.seed(self.simSeed)
            self.sim = simulator.Simulator(N=self.N, mean=self.mean, dt=self.dt*self.mins2days, rms=self.rms)
            #print(self.sim.dt)
            self.freq = np.fft.rfftfreq(self.sim.N, d=self.sim.dt)[1:]   # [1:] to remove the DC power in the spectrum
            spectrum = self.powmodel(self.freq, *self.powmodelparams)

            fig, ax = plt.subplots(1,2, figsize=[14,6])
            #lc = self.sim.simulate(spectrum)
            #plt.plot(lc.time/365, lc.counts, 'g')
            
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            #ax[0].set_title('Power spectrum', fontsize=16)
            ax[0].set_ylabel(r'$\nu P_{\nu}$', fontsize=14)
            ax[0].set_xlabel(r'$\nu\;[days^{-1}]$', fontsize=12)
            ax[0].grid()
            ax_top = ax[0].twiny()  
            ax_top.set_xlabel('timescale [days]', fontsize = 12) 
            ax_top.plot(1/self.freq, self.freq * spectrum,'r-', lw=3, alpha=0.0)
            ax_top.set_xscale('log')
            ax_top.set_yscale('log')
            for val in [0.5, 1, 3, 9, 18]:
                self.plot_vertical_line(ax_top, val*365, str(val)+' years')
            for val in [10, 100]:
                self.plot_vertical_line(ax_top, val, str(val)+' days')
            
            ax[0].plot(self.freq, self.freq * spectrum,'k-', lw=3, zorder=3)
            
            ax_top.tick_params(axis='x')
            ax_top.invert_xaxis()
            
            #ax_top.set_xticks(np.array(10.0**np.arange(1, -5, -0.5)).astype('float'))
            #ax_top.set_xticklabels(np.round(np.array(10.0**np.arange(1, -5, -0.5)), decimals=2).astype('str'))
            
            #ax_top.set_xticklabels(1/self.freq[(np.logspace(0, np.log10(len(self.freq))-1,10)).astype('int')])

            
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            #ax[1].set_title('Power spectrum', fontsize=16)
            ax[1].set_ylabel(r'$P_{\nu}$', fontsize=14)
            ax[1].set_xlabel(r'$\nu\;[days^{-1}]$', fontsize=12)
            ax[1].grid()
            ax_top = ax[1].twiny()  
            ax_top.set_xlabel('timescale [years]' , fontsize = 12) 
            ax_top.plot(1/self.freq/365, spectrum,'r-', lw=3, alpha=0.0)
            ax_top.set_xscale('log')
            ax_top.set_yscale('log')
            for val in [0.5, 1, 3, 9, 18]:
                self.plot_vertical_line(ax_top, val, str(val)+' years')
            for val in [10, 100]:
                self.plot_vertical_line(ax_top, val/365, str(val)+' days')
            
            ax_top.tick_params(axis='x')
            ax_top.invert_xaxis()
            ax[1].plot(self.freq, spectrum,'k-', lw=3, zorder=3)
            
            plt.tight_layout()
            plt.show()



    def LCsimulator(self):
        # This part of the code simulates the lightcurve using the already establised parameters and power spectrum
        np.random.seed(self.simSeed)
        self.sim = simulator.Simulator(N=self.N, mean=self.mean, dt=self.dt*self.mins2days, rms=self.rms)
        self.freq = np.fft.rfftfreq(self.sim.N, d=self.sim.dt)[1:]   # [1:] to remove the DC power in the spectrum

        spectrum = self.powmodel(self.freq, *self.powmodelparams)
        self.lc = self.sim.simulate(spectrum)


    def sort_arrays_together(self, time, flux):
        """
        Sorts two arrays (time and flux) together based on the values of the time array.

        Parameters:
        - time: array-like, array to be sorted
        - flux: array-like, corresponding flux values

        Returns:
        - sorted_time: array-like, sorted time array
        - sorted_flux: array-like, corresponding flux values sorted based on time
        """
        indices = np.argsort(time)
        sorted_time = time[indices]
        sorted_flux = flux[indices]
        return sorted_time, sorted_flux
    
    def sampler(self, OBSpNight = 3, NightsperWINDOW = 7, NumofWINDOW = 9 , OBSperiod = 8, WINDOWwidth = 10, dataLOSSfrac = 0, noiseSIGMA = 0, samplerSEED = 100, plot=True, returnVAL =True ):
        '''
        Function to sample using a simpler algorithm, Hence faster
        
        Parameters:
        -----------------------------
        OBSpNight       : Number of observations per night (NOT USING THIS NOW)
        NightsperWINDOW : Number of Nights per window that is observed
        NumofWINDOW     : Number of windows observed
        OBSperiod       : (IN MONTHS) The period of the sampling pattern
        WINDOWwidth     : (IN DAYS) Width of the window centered around the sampling pattern points
        dataLOSSfrac    : The fraction of data loss due to whether or any reasons randomly choosen
        noiseSIGMA      : sigma of the gaussian noise added
        samplerSEED     : Seed for the sampling algorithm
                
        '''

        if isinstance(self.lc, int):
            if self.verbose:
                print('Simulating lightcurve...')
            self.LCsimulator()
        else:
            if self.verbose:
                print('Loading existing simulation.. ')
                
        # This part samples the simulated lightcurve, from the sampling parameters provided here.
        np.random.seed(samplerSEED)

        # Selects the starting point for the simulated spectra randomly, by considering the provided total sampling length and 
        # the total length of the lightcurve
        if (OBSperiod*self.OneMonthIndex)>=len(self.lc.time):
            print('The given set of sampling conditions is not feasible with the light curve parameters provided')
            return 0
        elif (OBSperiod*30)<WINDOWwidth:
            print('Windows are overlapping. Try a different combination of OBSperiod and WINDOWwidth')
            return 0
        else:
            simstart_index = np.random.randint(0, len(self.lc.time)-NumofWINDOW*OBSperiod*self.OneMonthIndex)
            
        # Calculate the sampling pattern indices
        #print(simstart_index, (NumofWINDOW+1)*OBSperiod*self.OneMonthIndex)
        samplingPatternIndices = np.arange(simstart_index, simstart_index+(NumofWINDOW)*OBSperiod*self.OneMonthIndex, OBSperiod*self.OneMonthIndex)

        #print(samplingPatternIndices)
        # select the nights from the sampling pattern indices     
        samplingNightsIndices = []
        for val in samplingPatternIndices:
            samplingNightsIndices.append(np.random.choice(np.arange((val - (WINDOWwidth*self.OneMonthIndex/30)/2).astype('int'),
                                                            (val + (WINDOWwidth*self.OneMonthIndex/30)/2).astype('int')), size=NightsperWINDOW, replace=False))
        samplingNightsIndices = np.array(samplingNightsIndices).astype('int')
           
        # samplingNightsIndices = np.random.randint((samplingPatternIndices-(WINDOWwidth*self.OneMonthIndex/30)/2).astype('int')
        #                                             ,(samplingPatternIndices+(WINDOWwidth*self.OneMonthIndex/30)/2).astype('int'),
        #                                             size= [NightsperWINDOW, NumofWINDOW])
        
        simulationIndices = np.array(samplingNightsIndices).flatten()

        dataLOSSnumber = int(len(simulationIndices)*dataLOSSfrac)

        simulationIndices = np.random.choice(simulationIndices, size= len(simulationIndices)-dataLOSSnumber, replace=False)
        
        
        # Noise and decimal rounding in line with Vaughan et al 2016
        simTIME = self.lc.time[simulationIndices]/365
        simFLUX = np.round(self.lc.counts[simulationIndices] + np.random.normal(0, noiseSIGMA, size = len(simulationIndices)), decimals=6)
        simFLUXerr = np.ones(len(simTIME))*noiseSIGMA
        simTIME, simFLUX = self.sort_arrays_together(simTIME, simFLUX)

        
        if plot:
            plt.figure(figsize=[14,6])
            plt.plot(self.lc.time/365, self.lc.counts, 'k', alpha=0.5)
            
            plt.title('Simulated data', fontsize=16)
            plt.ylabel('Normalised Flux', fontsize=12)
            plt.xlabel('Time (years)', fontsize=12)
            
            plt.errorbar(simTIME, simFLUX, simFLUXerr, fmt='.b')
            plt.grid()
            
            ax_top = plt.gca().twiny()  
            ax_top.set_xlabel('Time (days)' , fontsize = 12) 
            ax_top.plot(self.lc.time,  self.lc.counts,'r-', lw=3, alpha=0.0)
            ax_top.tick_params(axis='x')
            
            
            plt.tight_layout()
            plt.savefig('plots/lightcruve.png')
            plt.show()
            
            self.recon_powerSpectra(simFLUX, simTIME, OBSperiod)
        
        if returnVAL:
            return simTIME, simFLUX, simFLUXerr, self.lc
        
    
    def moving_average_interpolated(self, data, window_size):
        moving_averages = []

        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            average = sum(window) / window_size
            moving_averages.append(average)

        # Interpolate to match the input data size
        interpolated_moving_averages = np.interp(
            range(len(data)),
            range(window_size - 1, len(data)),
            moving_averages
        )

        return interpolated_moving_averages
    
    def recon_powerSpectra(self, FLUX, TIME, OBSper=-1, dt= -1):
        fft_result = np.fft.fft(FLUX, len(TIME))
        TIME = TIME* 365 # time conversion to days
        if dt == -1:
            dt = (TIME[1]-TIME[0])
        else:
            pass
        

        plot_freq, plot_spec = LombScargle(TIME, FLUX).autopower(nyquist_factor=3)
        
        # frequencies = np.fft.rfftfreq(len(fft_result), d=  dt)  
        # power_spectrum = np.abs(fft_result) ** 2 
        # plot_freq =frequencies[0:len(frequencies)//2]
        # plot_spec = power_spectrum[0:len(frequencies)//2]    
        fig, ax = plt.subplots(1,2 ,figsize=(14, 6))
        
        
        ax[0].plot(plot_freq, savgol_filter(plot_freq*plot_spec, 1, 0), 'k.-', lw=0.5)
        ax[0].plot(plot_freq, self.moving_average_interpolated(plot_freq*plot_spec,50), 'r-' )
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        #ax[0].set_title('Power Spectrum')
        ax[0].set_xlabel(r'$\nu$',fontsize=16)
        ax[0].set_ylabel(r'$\nu P_{\nu}$',fontsize=16)
        
        #self.plot_vertical_line(ax[0], OBSper*30, 'Sampling period')
        ax[0].grid(True)       
        
        for val in [0.5, 1, 3, 9]:
            self.plot_vertical_line(ax[0], 1/(val*365), str(val)+' years')
        for val in [10, 100]:
            self.plot_vertical_line(ax[0], 1/val, str(val)+' days')
        if OBSper!=-1:
            self.plot_vertical_line(ax[0], 1/(OBSper*30), 'Sampling period',c='blue', ypos=0.5,boxalpha = 0.9, textalpha=1)
        
        ax_top = ax[0].twiny()  
        ax_top.set_xlabel('timescale [days]', fontsize = 12) 
        ax_top.plot(1/plot_freq, savgol_filter(plot_freq*plot_spec, 1, 0),'r-', lw=3, alpha=0.0)
        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.tick_params(axis='x')
        ax_top.invert_xaxis()
        
        
        ax[1].plot(plot_freq,savgol_filter(plot_spec, 1, 0), 'k.-', lw=0.5)
        ax[1].plot(plot_freq, self.moving_average_interpolated(plot_spec,50), 'r-' )
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_title('Power Spectrum')
        ax[1].set_xlabel(r'$\nu$',fontsize=16)
        ax[1].set_ylabel(r'$P_{\nu}$',fontsize=16)
        ax[1].grid(True)
        for val in [0.5, 1, 3, 9]:
            self.plot_vertical_line(ax[1], 1/(val*365), str(val)+' years')
        for val in [10, 100]:
            self.plot_vertical_line(ax[1], 1/val, str(val)+' days')
        self.plot_vertical_line(ax[1], 1/(OBSper*30), 'Sampling period',c='blue', ypos=0.5,boxalpha = 0.9, textalpha=1)
        
        ax_top = ax[1].twiny()  
        ax_top.set_xlabel('timescale [days]', fontsize = 12) 
        ax_top.plot(1/plot_freq, plot_spec,'r-', lw=3, alpha=0.0)
        ax_top.set_xscale('log')
        ax_top.set_yscale('log')
        ax_top.tick_params(axis='x')
        ax_top.invert_xaxis()
        
        
        plt.tight_layout()
        plt.show()
        
    
    
    def monsterr(self, OBSpNight = 6, NightsNum = 7, WindowNum = 9, windowLength = 4, gaplength = 6, samplerSEED = 100, dataLOSSfrac = 0, noiseSIGMA = 0.015 ,plot= True, returnVALS=True):
        '''
        Finallyyyyy, the function to sample the lightcurve aka ' de monsterr '

        Parameters:
        -----------------------------

        OBSpNight       : Number of observations per night
        NightsNum       : Number of Nights per season that is observed
        Windowlength    : Number of months in a single Observation window/season
        gaplength       : Number of months in between each observation window/season
        samplerSEED     : Seperate seed for the sampling part to allow, different sampling for the same lightcurve.

        '''
        start = time.time()
        # This part of the code simulates the lightcurve using the already establised parameters and power spectrum
        if isinstance(self.lc, int):
            if self.verbose:
                print('Simulating lightcurve...')
            self.LCsimulator()
        else:
            if self.verbose:
                print('Loading existing simulation.. ')
                
        # This part samples the simulated lightcurve, from the sampling parameters provided here.
        np.random.seed(samplerSEED)
        step1 = time.time()
        # Selects the starting point for the simulated spectra randomly, by considering the provided total sampling length and 
        # the total length of the lightcurve
        if ((windowLength + gaplength)*self.OneMonthIndex*WindowNum - gaplength*self.OneMonthIndex)>=len(self.lc.time):
            print('The given set of sampling conditions is not feasible with the light curve parameters provided')
            return 0
        else:
            simstart_index = np.random.randint(0, len(self.lc.time)-((windowLength + gaplength)*self.OneMonthIndex*WindowNum - gaplength*self.OneMonthIndex ))

        # Calculates the boundaries of each observation window/season
        OBSwindwIndices = []
        for i in range(WindowNum):
            OBSwindwIndices.append([int(simstart_index+(windowLength+gaplength)*self.OneMonthIndex*i), 
                                    int(simstart_index+ windowLength*self.OneMonthIndex +(windowLength+gaplength)*self.OneMonthIndex*i )])


        # Selects the nights in the observation window/season where observation was made
        NightsIndices = []
        for i, inds in enumerate(OBSwindwIndices):
            NightsIndices.append(np.random.choice(np.arange(inds[0], inds[1]+1,1),size=NightsNum, replace=False))
        NightsIndices = np.array(NightsIndices).flatten()
        
        # Selects the final data points from the lightcurve from the simulation such that observations fall after
        # 21:00 and before 5:00 everyday.
        simulationIndices = []
        for i, val  in enumerate(NightsIndices):
            astroNightIndices = np.argwhere((self.lc.time.astype('int')==int(self.lc.time[val])) & (((self.lc.time%1)<(5/24)) | ((self.lc.time%1)>(21/24)))).flatten()
            simulationIndices.append(np.random.choice(astroNightIndices, size=OBSpNight, replace=False ))

        simulationIndices = np.array(simulationIndices).flatten()

        dataLOSSnumber = int(len(simulationIndices)*dataLOSSfrac)

        simulationIndices = np.random.choice(simulationIndices, size= len(simulationIndices)-dataLOSSnumber, replace=False)
        
        step2 = time.time()
        
        # Noise and decimal rounding in line with Vaughan et al 2016
        simTIME = self.lc.time[simulationIndices]/365
        simFLUX = np.round(self.lc.counts[simulationIndices] + np.random.normal(0, noiseSIGMA, size = len(simulationIndices)), decimals=6)
        simFLUXerr = np.abs(np.random.normal(0, 4*noiseSIGMA, size= len(simTIME)))

        print(step2-step1, step1-start)
        
        if plot:
            plt.figure(figsize=[14,6])
            plt.plot(self.lc.time/365, self.lc.counts, 'k', alpha=0.5)
            
            plt.title('Simulated data', fontsize='16')
            plt.ylabel('Normalised Flux', fontsize='14')
            plt.xlabel('Time (years)', fontsize='14')
            #for i, ind in enumerate(OBSwindwIndices):
            #    plt.plot(self.lc.time[ind[0]:ind[1]], self.lc.counts[ind[0]:ind[1]], 'r-')

            #for i, val in enumerate(NightsIndices):
            #    plt.plot(self.lc.time[val], self.lc.counts[val], 'b.')

            #plt.plot(self.lc.time[simulationIndices], self.lc.counts[simulationIndices], 'b.')
            plt.errorbar(simTIME, simFLUX, simFLUXerr, fmt='.b')

            plt.grid()
            plt.tight_layout()
            plt.show()
        
        if returnVALS:
            return simTIME, simFLUX, simFLUXerr, self.lc
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class   JAXNSmodelCreator:
    '''
    A class to create the JAXNS model given the the generative_prior function and loglikelihood function with
    the flux, flux_err and time values
    
    '''
    def __init__(self, time, flux, flux_err):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err

    def _likelihood_ASSIGNER(self, input_function,  flux , time, flux_err, **kwargs):
        n_comp = 0
        kwargs_len = len(list(kwargs.keys()))
        if kwargs_len!=0:
            for key, value in kwargs.items():
                if key=='n_component':
                    n_comp = value
        def generated_function(*args, **kwargs):

            if (n_comp==0):
                updated_kwargs = {**kwargs, 'y': flux, 'y_errs': flux_err, 'time': time}
            else:
                updated_kwargs = {**kwargs, 'n_component':n_comp, 'y': flux, 'y_errs': flux_err, 'time': time}
            
            return input_function(*args, **updated_kwargs)

        return generated_function
    
    def _prior_ASSIGNER(self, input_function, time,):
        def generated_function(*args, **kwargs):

            updated_kwargs = {**kwargs, 'time': time}

            return input_function(*args, **updated_kwargs)

        return generated_function




    def create_model(self, prior_mod, log_likeli_fun, **kwargs):
        
        if len(list(kwargs.keys()))!=0:
            for key, value in kwargs.items():
                if key == 'n_component':
                    log_likeli_fun_updated = self._likelihood_ASSIGNER(log_likeli_fun, flux=self.flux, time= self.time, flux_err=self.flux_err, n_component = value )
                else:
                    log_likeli_fun_updated = self._likelihood_ASSIGNER(log_likeli_fun, flux=self.flux, time= self.time, flux_err=self.flux_err )
        else:
            log_likeli_fun_updated = self._likelihood_ASSIGNER(log_likeli_fun, flux=self.flux, time= self.time, flux_err=self.flux_err )
                    
        prior_mod_updated = self._prior_ASSIGNER( prior_mod, time = self.time)
        return jaxns.Model(prior_model=prior_mod_updated, log_likelihood=log_likeli_fun_updated)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
