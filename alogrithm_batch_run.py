import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.THESIS import LightCurveSampler, ModelComparison, JAXNSmodelCreator
from utils.NSmodels2 import *
import json, gc
import sys, os

def sort_arrays_together(time, flux, err=[None]):
        """
        Sorts two arrays (time and flux) together based on the values of the time array.

        """
        indices = np.argsort(time)
        sorted_time = time[indices]
        sorted_flux = flux[indices]
        if err[0]!=None:
            sorted_flux_err = err[indices]
            return sorted_time, sorted_flux, sorted_flux_err
        else:
            return sorted_time, sorted_flux
        
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def sample_posterior_within_1sigma(param_samples, num_samples):
    lower_quantiles, upper_quantiles = np.percentile(param_samples, [16, 84], axis=0)
    
    within_1sigma_mask = np.all((param_samples >= lower_quantiles) & (param_samples <= upper_quantiles), axis=1)
    filtered_samples = param_samples[within_1sigma_mask]
    
    if len(filtered_samples) < num_samples:
        raise ValueError("Not enough samples within 1$\sigma$ to draw the desired number of samples")
    sampled_indices = np.random.choice(filtered_samples.shape[0], size=num_samples, replace=False)
    sampled_params = filtered_samples[sampled_indices]
    
    return sampled_params




def orginal_data_run():
    modelCreater = JAXNSmodelCreator(time, flux, fluxerr)

    DRW_NSmodel = modelCreater.create_model(DRW_generative_prior, DRW_log_likelihood_model)
    DRW_sine_NSmodel = modelCreater.create_model(DRW_sine_generative_prior, DRW_sine_log_likelihood_model)

    CARMA21_NSmodel = modelCreater.create_model(create_CARMA_JAXNS_model_funcs(2,1, functype="prior"), 
                                                create_CARMA_JAXNS_model_funcs(2,1, functype="likelihood"))
    CARMA21_sine_NSmodel = modelCreater.create_model(create_CARMAsine_JAXNS_model_funcs(2,1, functype='prior'),
                                                create_CARMAsine_JAXNS_model_funcs(2,1, functype='likelihood'))
        
    OBPL_10_NSmodel = modelCreater.create_model(OBPL_generative_prior, OBPL_log_likelihood_model, n_component = 10)
    OBPLsine_10_NSmodel = modelCreater.create_model(OBPLsine_generative_prior, OBPLsine_log_likelihood_model, n_component = 10)

    print('generating posteriors for OBPL parameters for given data...', flush=True)
    blockPrint()
    ModelCOMP = ModelComparison(time, flux, fluxerr, verbose=True)
    ModelCOMP.add_NestedModel(OBPL_10_NSmodel, 'OBPL10')
    ModelCOMP.add_NestedModel(OBPLsine_10_NSmodel, 'OBPL10sine')

    ModelCOMP.add_NestedModel(DRW_NSmodel, 'DRW')
    ModelCOMP.add_NestedModel(DRW_sine_NSmodel, 'DRWsine')

    ModelCOMP.add_NestedModel(CARMA21_NSmodel, 'CARMA21')
    ModelCOMP.add_NestedModel(CARMA21_sine_NSmodel, 'CARMA21sine')
    ModelCOMP.run_NSsampler()
    enablePrint()
    print('Posteriors generated...', flush=True)


    # OBPL_dict = ModelCOMP.return_samples_logZ_dict('OBPL10')
    # OBPLsine_dict = ModelCOMP.return_samples_logZ_dict('OBPL10sine')
    # org_data_bayes = np.exp(OBPL_dict['log_Z_mean'] - OBPLsine_dict['log_Z_mean'])
    for i, val in enumerate(ModelCOMP.models()):
        model_dict = ModelCOMP.return_samples_logZ_dict(val)
        with open( save_file_path +  "original_data_" + val+ ".json", "w") as outfile: 
            json.dump(model_dict, outfile)
        del model_dict


    del ModelCOMP, modelCreater






if len(sys.argv) > 1:
    data_file_path = sys.argv[1]
    save_file_path = sys.argv[2]
    index1 = int(sys.argv[3])
    # index2 = int(sys.argv[4])

if os.path.isfile(data_file_path):
    print('data file identified', flush=True)
else :
    sys.exit()
    
if os.path.isdir(save_file_path):
    print('Destination folder identified', flush=True)
else :
    sys.exit()


time, flux, fluxerr = np.loadtxt(data_file_path).T





if __name__ == '__main__':
    
    
    # check if dict is already present 
    if os.path.isfile("/home/14444429/results/json_files/Graham2015/original_data_OBPL10.json"):
        print(1)
        with open("/home/14444429/results/json_files/Graham2015/original_data_OBPL10.json") as json_file:
            OBPL_dict = json.load(json_file)
    else:
        print(2)
        orginal_data_run()
        with open("/home/14444429/results/json_files/Graham2015/original_data_OBPL10.json") as json_file:
            OBPL_dict = json.load(json_file)
        
        
    mat = sample_posterior_within_1sigma(np.array(list(OBPL_dict.values())[2:]).T, num_samples=100)

    alpha_h_array = mat.T[0]
    alpha_l_array = mat.T[1]
    err_scale_array = mat.T[2]
    log_bend_freq_array = mat.T[3]
    log_norm_array = mat.T[4]

    # posterior_dataset = pd.DataFrame(mat, columns= ['alpha_h', 'alpha_l', 'err_scale', 'log_bend_freq', 'log_norm'])

    logZ_OBPL = []
    logZ_OBPLsine = []
    bayes_OBPL_OBPLsine = []
    print('False positivity test initiated...', flush=True)

    for j in range(1):
        i = j + index1
        print(i, end='-->', flush=True)
        psd = OneBendPowerLaw([np.power(10, log_norm_array[i]), alpha_l_array[i], 10**(log_bend_freq_array[i]), alpha_h_array[i]], free_parameters=[True]*4)
        model_ACV = PSDToACV(psd, S_low=100,
                    S_high=20,
                    T=time[-1] - time[0],
                    dt=np.min(np.diff(time)),
                    method='SHO',
                    n_components=10,
                    estimate_variance=True,
                    init_variance=jnp.var(flux, ddof=1),
                    use_celerite=False,
                    use_legacy_celerite=False )

        kernel = model_ACV.ACVF
        gp = GaussianProcess(kernel, jnp.array(time), diag = jnp.array((err_scale_array[i]*fluxerr)**2), mean = jnp.array(0.0))
        y_gen_time = np.array(time).copy()
        y_gen = gp.sample(jax.random.PRNGKey(np.random.randint(0, 10000)))
        y_gen_err =  np.random.normal(np.mean(fluxerr), np.std(fluxerr))
        
        modelCreater = JAXNSmodelCreator(y_gen_time, y_gen, y_gen_err)

        DRW_NSmodel = modelCreater.create_model(DRW_generative_prior, DRW_log_likelihood_model)
        DRW_sine_NSmodel = modelCreater.create_model(DRW_sine_generative_prior, DRW_sine_log_likelihood_model)

        CARMA21_NSmodel = modelCreater.create_model(create_CARMA_JAXNS_model_funcs(2,1, functype="prior"), 
                                                    create_CARMA_JAXNS_model_funcs(2,1, functype="likelihood"))
        CARMA21_sine_NSmodel = modelCreater.create_model(create_CARMAsine_JAXNS_model_funcs(2,1, functype='prior'),
                                                    create_CARMAsine_JAXNS_model_funcs(2,1, functype='likelihood'))
        
        OBPL_10_NSmodel = modelCreater.create_model(OBPL_generative_prior, OBPL_log_likelihood_model, n_component = 10)
        OBPLsine_10_NSmodel = modelCreater.create_model(OBPLsine_generative_prior, OBPLsine_log_likelihood_model, n_component = 10)
        
        blockPrint()
        ModelCOMP = ModelComparison(y_gen_time, y_gen, y_gen_err, verbose=True)
        ModelCOMP.add_NestedModel(OBPL_10_NSmodel, 'OBPL10')
        ModelCOMP.add_NestedModel(OBPLsine_10_NSmodel, 'OBPL10sine')
            
        ModelCOMP.add_NestedModel(DRW_NSmodel, 'DRW')
        ModelCOMP.add_NestedModel(DRW_sine_NSmodel, 'DRWsine')

        ModelCOMP.add_NestedModel(CARMA21_NSmodel, 'CARMA21')
        ModelCOMP.add_NestedModel(CARMA21_sine_NSmodel, 'CARMA21sine')
            
        ModelCOMP.run_NSsampler()
        enablePrint()
        OBPL_dict = ModelCOMP.return_samples_logZ_dict('OBPL10')
        OBPLsine_dict = ModelCOMP.return_samples_logZ_dict('OBPL10sine')
        
        # logZ_OBPL.append(OBPL_dict['log_Z_mean'])
        # logZ_OBPLsine.append(OBPLsine_dict['log_Z_mean'])
        # bayes_OBPL_OBPLsine.append(np.exp(logZ_OBPL-logZ_OBPLsine))
        for i, val in enumerate(ModelCOMP.models()):
                model_dict = ModelCOMP.return_samples_logZ_dict(val)
                with open( save_file_path + str(int(index1)) + "_" + val+ ".json", "w") as outfile: 
                    json.dump(model_dict, outfile)
                del model_dict
        
        del modelCreater,ModelCOMP
        del   OBPL_10_NSmodel, OBPLsine_10_NSmodel
            
        gc.collect()
        jax.clear_caches()