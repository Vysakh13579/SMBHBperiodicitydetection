import os
#import multiprocessing

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax

#print(os.environ["XLA_FLAGS"])
#jax.distributed.initialize(coordinator_address=)

import pandas as pd
jax.config.update("jax_enable_x64", True)

from THESIS import LightCurveSampler, JAXNSmodelCreator, ModelComparison
from NSmodels_noerrscale import *
import warnings
warnings.filterwarnings('ignore')
import json
import gc
import numpy as np
# import time
# import psutil
import os
import sys

if len(sys.argv) > 1:
    save_json_folder_path = sys.argv[2]
    csv_file_path = sys.argv[1]
    index1 = int(sys.argv[3])
    # index2 = int(sys.argv[4])
else:
    csv_file_path = 'simDATAcsvs/simDATA_highALPHA_3_NumofWINDOW_4_20.csv' # the csv folder
    save_json_folder_path = 'test_results/'    # Folder to save files
    index1 = 3
    #index2 = -1


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


simDATA = pd.read_csv(csv_file_path)

print("dataframe index :", len(simDATA) - int(index1)-1, flush=True)
#print("index 2 :", index2, flush=True)

for r, row in simDATA.sort_index(ascending=False)[index1:index1+1].iterrows():             

    #try:
        #
        l = LightCurveSampler(N=2**21, rms=row.rms, simulatorSEED= int(row.simSEED), verbose=False)
        l.load_powerspec(bend_pl, [200,  row.bendfreq, 
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
        
        del l
        
        lcTIME = simTIME - np.mean(simTIME)
        lcFLUX = simLC - np.median(simLC)
        lcFLUXerr = simLCerr
        #print(r, end='-->', flush=True)
        
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
        
        ModelCOMP.run_NSsampler(name= 'DRW', num_par_samplers=1)
        print("Finished DRW..", flush=True)

        ModelCOMP.run_NSsampler(name= 'DRWsine', num_par_samplers=1)
        print("Finished DRWsine..", flush=True)

        ModelCOMP.run_NSsampler(name= 'CARMA21', num_par_samplers=1)
        print("Finished CARAM21..", flush=True)
        
        ModelCOMP.run_NSsampler(name= 'CARMA21sine', num_par_samplers=1)
        print("Finished CARMA21sine..", flush=True)
        
        ModelCOMP.run_NSsampler(name= 'OBPL10', num_par_samplers=1)
        print("Finished OBPL10..", flush=True)
        
        ModelCOMP.run_NSsampler(name= 'OBPLsine10', num_par_samplers=1)
        print("Finished OBPL10sine..", flush=True)
        
        # if index1%6==0:
        #     print("Model used : DRW", flush=True)
        #     DRW_NSmodel = modelCreater.create_model(DRW_generative_prior, DRW_log_likelihood_model)
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(DRW_NSmodel, 'DRW')
        #     ModelCOMP.run_NSsampler(name= 'DRW', num_par_samplers=1)
        
        # elif index1%6==1:
        #     print("Model used : DRW + sine", flush=True)
        #     DRW_sine_NSmodel = modelCreater.create_model(DRW_sine_generative_prior, DRW_sine_log_likelihood_model)
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(DRW_sine_NSmodel, 'DRWsine')
        #     ModelCOMP.run_NSsampler(name= 'DRWsine', num_par_samplers=1)
        
        # elif index1%6==2:
        #     print("Model used : CARMA(2,1)", flush=True)
        #     CARMA21_NSmodel = modelCreater.create_model(create_CARMA_JAXNS_model_funcs(2,1, functype="prior"), 
        #                                            create_CARMA_JAXNS_model_funcs(2,1, functype="likelihood"))
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(CARMA21_NSmodel, 'CARMA21')
        #     ModelCOMP.run_NSsampler(name= 'CARMA21', num_par_samplers=1)
        
        # elif index1%6==3:
        #     print("Model used : CARMA(2,1) + sine", flush=True)
        #     CARMA21_sine_NSmodel = modelCreater.create_model(create_CARMAsine_JAXNS_model_funcs(2,1, functype='prior'),
        #                                            create_CARMAsine_JAXNS_model_funcs(2,1, functype='likelihood'))
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(CARMA21_sine_NSmodel, 'CARMA21sine')
        #     ModelCOMP.run_NSsampler(name= 'CARMA21sine', num_par_samplers=1)
        
        # elif index1%6==4:
        #     print("Model used : OBPL 10 comp", flush=True)
        #     OBPL_10_NSmodel = modelCreater.create_model(OBPL_generative_prior, OBPL_log_likelihood_model, n_component = 10)
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(OBPL_10_NSmodel, 'OBPL10')
        #     ModelCOMP.run_NSsampler(name= 'OBPL10', num_par_samplers=1)
        
        # elif index1%6==5:
        #     print("Model used : OBPL 10 comp + sine", flush=True)
        #     OBPLsine_10_NSmodel = modelCreater.create_model(OBPLsine_generative_prior, OBPLsine_log_likelihood_model, n_component = 10)
        #     ModelCOMP = ModelComparison(lcTIME, lcFLUX,lcFLUXerr, verbose=False)
        #     ModelCOMP.add_NestedModel(OBPLsine_10_NSmodel, 'OBPLsine10')
        #     ModelCOMP.run_NSsampler(name= 'OBPLsine10', num_par_samplers=1)
            
        
        
        
        for i, val in enumerate(ModelCOMP.models()):
            model_dict = ModelCOMP.return_samples_logZ_dict(val)
            with open( save_json_folder_path + str(int(row.ID)) + "_" + val+ ".json", "w") as outfile: 
                json.dump(model_dict, outfile)
            del model_dict
        
        
        
        del lcTIME, lcFLUX, lcFLUXerr, lc , modelCreater, DRW_NSmodel, DRW_sine_NSmodel, ModelCOMP#, sine_NSmodel,
        del  simTIME, simLC, simLCerr , CARMA21_NSmodel, CARMA21_sine_NSmodel, OBPL_10_NSmodel, OBPLsine_10_NSmodel
        
        gc.collect()
        jax.clear_caches()
        

