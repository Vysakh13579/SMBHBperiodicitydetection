import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.transforms as transforms

from tinygp import kernels
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import jaxns


import tensorflow_probability.substrates.jax.distributions as tfpd

from functools import partial
import warnings
import gc
import json
warnings.filterwarnings('ignore')
import os, sys

from utils.THESIS import LightCurveSampler, ModelComparison, JAXNSmodelCreator
from utils.NSmodels import *

if len(sys.argv) > 1:
    save_json_folder_path = sys.argv[2]
    csv_file_path = sys.argv[1]
    index1 = int(sys.argv[3])
    index2 = int(sys.argv[4])
else:
    csv_file_path = 'simDATAcsvs/simDATA_highALPHA_3_NumofWINDOW_12_period_0.75_6_A1_0.015_0.3.csv' # the csv folder
    save_json_folder_path = 'NS_results/'    # Folder to save files
    index1 = 3
    index2 = -1


if os.path.isfile(csv_file_path):
    print('csv file identified')
else :
    sys.exit()
    
if os.path.isdir(save_json_folder_path):
    print('Destination folder identified')
else :
    sys.exit()

def bend_pl(f, norm, f_bend, alph_lo, alph_hi, sharpness):
    '''
    Function for bend power_law creation
    '''
    powmod = (norm*(f/f_bend)**alph_lo)/(1.+(f/f_bend)**(sharpness*(alph_lo-alph_hi)))**(1./sharpness)
    return powmod

def sin_curve(A, period, phase , time):
    return A * np.sin(2 * np.pi/ period * time + phase)


simDATA = pd.read_csv(csv_file_path)

print("sampling inititated", flush = True)
for r, row in simDATA[index1:index2].iterrows():             

    #try:
        l = LightCurveSampler(N=2**21, rms=row.rms, simulatorSEED= int(row.simSEED), mean = partial(sin_curve, 0.15*row.A1, row.period, 0) , verbose=False)
        l.load_powerspec(bend_pl, [20,  row.bendfreq, 
                                    row.lowalpha,
                                    row.highalpha,
                                    row.sharpness], 
                                    plot=False)
        l.LCsimulator()
        simTIME, simLC, simLCerr , lc = l.sampler(NightsperWINDOW = int(row.NightsperWINDOW), 
                                                    NumofWINDOW = int(row.NumofWINDOW),
                                                    OBSperiod = row.OBSperiod,
                                                    WINDOWwidth = row.WINDOWwidth,
                                                    samplerSEED = int(row.sampleSEED),
                                                    dataLOSSfrac = row.dataLOSSfrac, 
                                                    noiseSIGMA = row.noiseSIGMA, plot=False)
        
        lcTIME = simTIME
        lcFLUX = simLC - np.median(simLC)
        lcFLUXerr = simLCerr
        print(r, end='-->', flush=True)
        
        modelCreater = JAXNSmodelCreator(lcTIME, lcFLUX, lcFLUXerr)

        DRW_NSmodel = modelCreater.create_model(DRW_generative_prior, DRW_log_likelihood_model)
        DRW_sine_NSmodel = modelCreater.create_model(DRW_sine_generative_prior, DRW_sine_log_likelihood_model)
        
        CARMA21_NSmodel = modelCreater.create_model(create_CARMA_JAXNS_model_funcs(2,1, functype="prior"), 
                                                    create_CARMA_JAXNS_model_funcs(2,1, functype="likelihood"))
        CARMA21_sine_NSmodel = modelCreater.create_model(create_CARMAsine_JAXNS_model_funcs(2,1, functype='prior'),
                                                    create_CARMAsine_JAXNS_model_funcs(2,1, functype='likelihood'))
        
        OBPL_10_NSmodel = modelCreater.create_model(OBPL_generative_prior, OBPL_log_likelihood_model, n_component = 10)
        OBPLsine_10_NSmodel = modelCreater.create_model(OBPLsine_generative_prior, OBPLsine_log_likelihood_model, n_component = 10)
        
        ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        ModelCOMP.add_NestedModel(DRW_NSmodel, 'DRW')
        ModelCOMP.add_NestedModel(DRW_sine_NSmodel, 'DRWsine')

        ModelCOMP.add_NestedModel(CARMA21_NSmodel, 'CARMA21')
        ModelCOMP.add_NestedModel(CARMA21_sine_NSmodel, 'CARMA21sine')
        
        ModelCOMP.add_NestedModel(OBPL_10_NSmodel, 'OBPL10')
        ModelCOMP.add_NestedModel(OBPLsine_10_NSmodel, 'OBPLsine10')
        
        #ModelCOMP.run_NSsampler()
        
        ModelCOMP.run_NSsampler(name= 'DRW', num_par_samplers=1)
        ModelCOMP.run_NSsampler(name = 'DRWsine', num_par_samplers=1)
        ModelCOMP.run_NSsampler(name = 'CARMA21', num_par_samplers=1)
        ModelCOMP.run_NSsampler(name = 'CARMA21sine', num_par_samplers=1)
        ModelCOMP.run_NSsampler(name = 'OBPL10', num_par_samplers=1)
        ModelCOMP.run_NSsampler(name = 'OBPLsine10', num_par_samplers=1)
        
        for i, val in enumerate(ModelCOMP.models()):
            model_dict = ModelCOMP.return_samples_logZ_dict(val)
            with open( save_json_folder_path + str(int(row.ID)) + "_" + val+ ".json", "w") as outfile: 
                json.dump(model_dict, outfile)
            del model_dict
        
        
        
        del l, lcTIME, lcFLUX, lcFLUXerr, lc , modelCreater, DRW_NSmodel, DRW_sine_NSmodel, ModelCOMP#, sine_NSmodel,
        del  simTIME, simLC, simLCerr , CARMA21_sine_NSmodel, CARMA21_NSmodel, OBPL_10_NSmodel, OBPLsine_10_NSmodel
        
        gc.collect()
        jax.clear_caches()
        

