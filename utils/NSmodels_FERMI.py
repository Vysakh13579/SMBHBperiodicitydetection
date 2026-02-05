
import jaxns
import warnings
warnings.filterwarnings('ignore')
import tensorflow_probability.substrates.jax.distributions as tfpd
from tinygp import kernels
from tinygp import GaussianProcess
from functools import partial
import jax.numpy as jnp
import jax
import numpy as np
from pioran import PSDToACV
from pioran.psd import OneBendPowerLaw
jax.config.update("jax_enable_x64", True)






#DRW model prior
def DRW_generative_prior(time =None):
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0,1.5), name='err_scale')
    #m = yield jaxns.Prior(tfpd.Uniform(0,5), name='m')
    #b = yield jaxns.Prior(tfpd.Uniform(-10, 1), name='b')
    
    return log_amp, log_bend_freq, err_scale

# DRW model log likelihood model
def DRW_log_likelihood_model(log_amp, log_bend_freq, err_scale, time = None, y = None, y_errs = None ):
    
    kernel = jnp.power(10, log_bend_freq) * kernels.Exp(scale = 1/jnp.power(10, log_bend_freq))
    #mean_params = {'m': m, 'b':b}
    
    gp = GaussianProcess(kernel, time, diag=(err_scale*y_errs)**2, mean=0)
    
    return gp.log_probability(y)





# DRW + Linear model prior
def DRW_linear_generative_prior(time =None):
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0,1.5), name='err_scale')
    m = yield jaxns.Prior(tfpd.Uniform(-2,2), name='m')
    b = yield jaxns.Prior(tfpd.Uniform(-2, 2), name='b')
    
    return log_amp, log_bend_freq, m, b, err_scale

# linear model
def linear_model(params, time):
    
    return params['m']*time + params['b']
    

# DRW + linear model log likelihood model
def DRW_linear_log_likelihood_model(log_amp, log_bend_freq, m, b, err_scale, time = None, y = None, y_errs = None ):
    
    kernel = jnp.power(10, log_bend_freq) * kernels.Exp(scale = 1/jnp.power(10, log_bend_freq))
    mean_params = {'m': m, 'b':b}
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=partial(linear_model, mean_params))
    
    return gp.log_probability(y)



# Sine mean function model
def sine_model(A1, A2, t0, time = None ):
    
    return A1* jnp.cos(2*jnp.pi*time/t0) + A2* jnp.sin(2*jnp.pi*time/t0) 

# sine + linear mean function model
def sin_linear_model(A1, A2, t0, m, b, time=None):
    
    return sine_model(A1, A2, t0, time ) + m*time + b

# sine model prior
def sine_generative_prior():
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 1.67), name ='t0')
    nu = yield jaxns.Prior( tfpd.Uniform(0.0, 1.), name = 'nu')
    
    return A1, A2, t0, nu


# sine + linear model prior
def sine_linear_generative_prior():
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 1.67), name ='t0')
    nu = yield jaxns.Prior( tfpd.Uniform(0.0, 1.), name = 'nu')
    m = yield jaxns.Prior(tfpd.Uniform(0,5), name='m')
    b = yield jaxns.Prior(tfpd.Uniform(-2, 2), name='b')

    return A1, A2, t0, nu,  m, b,


# sine model log likelihood
def sine_log_likelihood_model(A1, A2, t0, nu, time = None, y = None, y_errs = None):
    
    loglikeli = - (y - sine_model(A1, A2, t0, time))**2/(2*(nu*y_errs)**2) - jnp.log(jnp.sqrt(2*jnp.pi)*y_errs*nu)
    return jnp.sum(loglikeli)

# sine + linear model log likelihood
def sine_linear_log_likelihood_model(A1, A2, t0, m, b, nu, time = None, y = None, y_errs = None):
    
    loglikeli = - (y - sin_linear_model(A1, A2, t0, m, b, time))**2/(2*(nu*y_errs)**2) - jnp.log(jnp.sqrt(2*jnp.pi)*y_errs*nu)
    return jnp.sum(loglikeli)



# DRW + sine model prior
def DRW_sine_generative_prior(time =None):
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0,1.5), name='err_scale')
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 8), name ='t0')
    #nu = yield jaxns.Prior( tfpd.Uniform(0.0, 1.), name = 'nu')
    
    return log_amp, log_bend_freq, err_scale, A1, A2, t0 #nu


# DRW + sine + linear model prior
def DRW_sine_linear_generative_prior(time=None):
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0,1.5), name='err_scale')
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 8), name ='t0')
    m = yield jaxns.Prior(tfpd.Uniform(-2,2), name='m')
    b = yield jaxns.Prior(tfpd.Uniform(-2, 2), name='b')
    
    return log_amp, log_bend_freq, A1, A2, t0, m, b, err_scale


# Sine curve mean function for DRW
def sine_curve_model(params, time):
    
    return params['A1']* jnp.cos(2*jnp.pi*time/params['t0']) + params['A2']* jnp.sin(2*jnp.pi*time/params['t0']) 

# Sine curve + linear mean function for DRW
def sine_linear_model(params, time):
    
    return sine_curve_model(params, time) + params['m']*time + params['b']

# DRW + sine + linear model loglikelihood 
def DRW_sine_linear_log_likelihood_model(log_amp, log_bend_freq, A1, A2, t0, m, b, err_scale, time = None, y = None, y_errs = None):
    
    kernel = jnp.power(10, log_amp) * kernels.Exp(scale = 1/jnp.power(10,log_bend_freq))
    
    mean_params = {'A1': A1, 'A2':A2, 't0':t0, 'm':m, 'b':b}
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean= partial(sine_linear_model, mean_params) )
    
    return gp.log_probability(y)

# DRW + sine  model loglikelihood 
def DRW_sine_log_likelihood_model(log_amp, log_bend_freq, err_scale, A1, A2, t0, time = None, y = None, y_errs = None):
    
    kernel = jnp.power(10, log_amp) * kernels.Exp(scale = 1/jnp.power(10,log_bend_freq))
    
    mean_params = {'A1': A1, 'A2':A2, 't0':t0}
    
    gp = GaussianProcess(kernel, time, diag=(err_scale*y_errs)**2, mean= partial(sine_curve_model, mean_params) )
    
    return gp.log_probability(y)


# CARMA(p,q) generative prior format function
def CARMA_gen_prior_dynamic_code(p, q):
    indent = "    "
    dynamic_code_str = "def CARMA"+str(p)+str(q)+"_generative_prior(time=None):" + "\n"
    
    # log_alpha codes
    for i in range(p):
        dynamic_code_str = dynamic_code_str + indent + "log_alpha"+str(i) + " = yield jaxns.Prior( tfpd.Uniform(-7,7), name = 'log_alpha"+str(i) + "')" + "\n"
    # log_beta codes
    for i in range(q):
        dynamic_code_str = dynamic_code_str + indent + "log_beta"+str(i+1) + " = yield jaxns.Prior( tfpd.Uniform(-15, 15), name = 'log_beta"+str(i+1) + "')" + "\n"
        
    dynamic_code_str =  dynamic_code_str + indent + "log_sigma = yield jaxns.Prior( tfpd.Uniform(-3, 5.3), name = 'log_sigma')" + "\n\n"
    dynamic_code_str =  dynamic_code_str + indent + "err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')" + "\n\n"
    
    dynamic_code_str =  dynamic_code_str + indent +  "return "
    
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "log_alpha"+str(i) + ", " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "log_beta"+str(i+1) + ", "
        
    dynamic_code_str = dynamic_code_str + "log_sigma, err_scale" 
    
    return dynamic_code_str

# CARMA(p,q) log likelihood format function
def CARMA__log_likelihood_dynamic_code(p, q):
    indent = "    "
    dynamic_code_str = "def CARMA"+str(p)+str(q)+"_log_likelihood_model("
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "log_alpha"+str(i) + ", " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "log_beta"+str(i+1) + ", "
        
    dynamic_code_str = dynamic_code_str + "log_sigma, err_scale, time = None, y = None, y_errs = None):" + "\n\n" 
    dynamic_code_str = dynamic_code_str + indent + "kernel = kernels.quasisep.CARMA.init("
    
    dynamic_code_str = dynamic_code_str + "alpha=["
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "jnp.exp(log_alpha"+str(i) + "), " 
    
    dynamic_code_str = dynamic_code_str + "], \n" + indent*8 +  " beta=[jnp.exp(log_sigma) * 1, " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "jnp.exp(log_sigma) * jnp.exp(log_beta"+str(i+1) + "), "
    
    dynamic_code_str = dynamic_code_str + "])" +  "\n\n"
    
    dynamic_code_str = dynamic_code_str + indent + "gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=0)" + "\n\n"
    
    dynamic_code_str = dynamic_code_str + indent + "return gp.log_probability(y)"
    
    return dynamic_code_str

# CARMA(p,q) function creators
def create_CARMA_JAXNS_model_funcs(p, q, functype):
    # Compile the dynamic code string into a code object
    
    if functype == "prior":
        dynamic_code = CARMA_gen_prior_dynamic_code(p, q)
        func_name = "CARMA"+str(p)+str(q)+"_generative_prior"
    elif functype == "likelihood":
        dynamic_code = CARMA__log_likelihood_dynamic_code(p,q)
        func_name = "CARMA"+str(p)+str(q)+"_log_likelihood_model"
    else:
        print("wrong functype!!")
    
    code_obj = compile(dynamic_code, '<string>', 'exec')

    # Create a namespace to hold the function
    namespace = {}
    
    # Execute the compiled code within the namespace
    exec(code_obj, globals(), namespace)

    # Extract the dynamically created function from the namespace
    dynamic_function = namespace[func_name]

    dynamic_function.__source__ = dynamic_code
    
    return dynamic_function


# CARMA(p,q) + sine generative prior format function
def CARMAsine_gen_prior_dynamic_code(p, q):
    indent = "    "
    dynamic_code_str = "def CARMA"+str(p)+str(q)+"_sine_generative_prior(time=None):" + "\n"
    
    # log_alpha codes
    for i in range(p):
        dynamic_code_str = dynamic_code_str + indent + "log_alpha"+str(i) + " = yield jaxns.Prior( tfpd.Uniform(-7,7), name = 'log_alpha"+str(i) + "')" + "\n"
    # log_beta codes
    for i in range(q):
        dynamic_code_str = dynamic_code_str + indent + "log_beta"+str(i+1) + " = yield jaxns.Prior( tfpd.Uniform(-15, 15), name = 'log_beta"+str(i+1) + "')" + "\n"
        
    dynamic_code_str =  dynamic_code_str + indent + "log_sigma = yield jaxns.Prior( tfpd.Uniform(-3, 5.3), name = 'log_sigma')" + "\n\n"
    dynamic_code_str =  dynamic_code_str + indent + "err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')" + "\n"
    
    dynamic_code_str =  dynamic_code_str + indent + "A1 = yield jaxns.Prior( tfpd.Uniform(0,0.3), name = 'A1') "+ "\n"
    #dynamic_code_str =  dynamic_code_str + indent + "phi = yield jaxns.Prior( tfpd.Normal(-np.pi,np.pi), name = 'phi') "+ "\n"
    dynamic_code_str =  dynamic_code_str + indent + "A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')"+ "\n"
    dynamic_code_str =  dynamic_code_str + indent + "t0 = yield jaxns.Prior( tfpd.Uniform(0, 8), name ='t0')"+ "\n\n"
    
    dynamic_code_str =  dynamic_code_str + indent +  "return "
    
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "log_alpha"+str(i) + ", " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "log_beta"+str(i+1) + ", "
        
    dynamic_code_str = dynamic_code_str + "log_sigma, A1, A2, t0, err_scale" 
    
    return dynamic_code_str
    
    
# CARMA(p,q)  sine log likelihood format function
def CARMAsine__log_likelihood_dynamic_code(p, q):
    indent = "    "
    dynamic_code_str = "def CARMA"+str(p)+str(q)+"_sine_log_likelihood_model("
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "log_alpha"+str(i) + ", " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "log_beta"+str(i+1) + ", "
        
    dynamic_code_str = dynamic_code_str + "log_sigma, A1, A2, t0, err_scale, time = None, y = None, y_errs = None):" + "\n\n" 
    dynamic_code_str = dynamic_code_str + indent + "kernel = kernels.quasisep.CARMA.init("
    
    dynamic_code_str = dynamic_code_str + "alpha=["
    for i in range(p):
        dynamic_code_str = dynamic_code_str + "jnp.exp(log_alpha"+str(i) + "), " 
    
    dynamic_code_str = dynamic_code_str + "], \n" + indent*8 +  " beta=[jnp.exp(log_sigma) * 1, " 
    for i in range(q):
        dynamic_code_str = dynamic_code_str + "jnp.exp(log_sigma) * jnp.exp(log_beta"+str(i+1) + "), "
    
    dynamic_code_str = dynamic_code_str + "])" +  "\n\n"
    
    dynamic_code_str = dynamic_code_str + indent + "mean_params = {'A1': A1, 'A2':A2, 't0':t0}"+  "\n\n"
    
    dynamic_code_str = dynamic_code_str + indent + "gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=partial(sine_curve_model, mean_params))" + "\n\n"
    
    dynamic_code_str = dynamic_code_str + indent + "return gp.log_probability(y)"
    
    return dynamic_code_str

# CARMA(p,q) + sine functin creators
def create_CARMAsine_JAXNS_model_funcs(p, q, functype):
    # Compile the dynamic code string into a code object
    
    if functype == "prior":
        dynamic_code = CARMAsine_gen_prior_dynamic_code(p, q)
        func_name = "CARMA"+str(p)+str(q)+"_sine_generative_prior"
    elif functype == "likelihood":
        dynamic_code = CARMAsine__log_likelihood_dynamic_code(p,q)
        func_name = "CARMA"+str(p)+str(q)+"_sine_log_likelihood_model"
    else:
        print("wrong functype!!")
    
    code_obj = compile(dynamic_code, '<string>', 'exec')

    # Create a namespace to hold the function
    namespace = {}
    
    # Execute the compiled code within the namespace
    exec(code_obj, globals(), namespace)

    # Extract the dynamically created function from the namespace
    dynamic_function = namespace[func_name]

    dynamic_function.__source__ = dynamic_code
    
    return dynamic_function




#One bend power law generative prior
def OBPL_generative_prior(time  = None):
    alpha_l = yield jaxns.Prior( tfpd.Uniform(0, 2), name = 'alpha_l')
    alpha_h = yield jaxns.Prior( tfpd.Uniform(alpha_l , 4), name = 'alpha_h')
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')
    
    return alpha_l, alpha_h, log_bend_freq, log_amp, err_scale

#One bend power law likelihood
def OBPL_log_likelihood_model(alpha_l, alpha_h, log_bend_freq, log_amp, err_scale,  n_component=10, time = None, y = None, y_errs = None ):
    
    psd = OneBendPowerLaw([jnp.power(10, log_amp), alpha_l, jnp.power(10, log_bend_freq), alpha_h], free_parameters=[True]*4)
    model_ACV = PSDToACV(psd, S_low=100,
                S_high=20,
                T=time[-1] - time[0],
                dt=np.min(np.diff(time)),
                method='SHO',
                n_components=n_component,
                estimate_variance=True,
                init_variance=jnp.var(y, ddof=1),
                use_celerite=False,
                use_legacy_celerite=False )
    
    kernel = model_ACV.ACVF
    
    
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=0)
    
    return gp.log_probability(y)


#One bend power law generative prior  linear
def OBPL_linear_generative_prior(time  = None):
    alpha_l = yield jaxns.Prior( tfpd.Uniform(-0.25, 2), name = 'alpha_l')
    alpha_h = yield jaxns.Prior( tfpd.Uniform(alpha_l , 4), name = 'alpha_h')
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')
    m = yield jaxns.Prior(tfpd.Uniform(-2,2), name='m')
    b = yield jaxns.Prior(tfpd.Uniform(-2, 2), name='b')
    
    return alpha_l, alpha_h, log_bend_freq, log_amp, err_scale, m, b

#One bend power law likelihood linear
def OBPL_linear_log_likelihood_model(alpha_l, alpha_h, log_bend_freq, log_amp, err_scale, m, b,  n_component=10, time = None, y = None, y_errs = None ):
    
    psd = OneBendPowerLaw([jnp.power(10, log_amp), alpha_l, jnp.power(10, log_bend_freq), alpha_h], free_parameters=[True]*4)
    model_ACV = PSDToACV(psd, S_low=100,
                S_high=20,
                T=time[-1] - time[0],
                dt=np.min(np.diff(time)),
                method='SHO',
                n_components=n_component,
                estimate_variance=True,
                init_variance=jnp.var(y, ddof=1),
                use_celerite=False,
                use_legacy_celerite=False )
    
    kernel = model_ACV.ACVF
    
    mean_params = {'m': m, 'b':b}
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=partial(linear_model, mean_params))
    
    return gp.log_probability(y)


#One bend power law + sine generative prior
def OBPLsine_generative_prior(time  = None):
    alpha_l = yield jaxns.Prior( tfpd.Uniform(-0.25, 2), name = 'alpha_l')
    alpha_h = yield jaxns.Prior( tfpd.Uniform(alpha_l , 4), name = 'alpha_h')
    # log_bend_freq = yield jaxns.Prior( tfpd.Uniform(np.log10(1/(time[-1]-time[0])),np.log10( 1/np.min(np.diff(time)))), name = 'log_bend_freq') # 1/years
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 8), name ='t0')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')
    
    return alpha_l, alpha_h, log_bend_freq, log_amp, A1, A2, t0, err_scale


#One bend power law + sine likelihood
def OBPLsine_log_likelihood_model(alpha_l, alpha_h, log_bend_freq, log_amp, A1, A2, t0, err_scale, n_component=10, time = None, y = None, y_errs = None ):
    
    psd = OneBendPowerLaw([jnp.power(10, log_amp), alpha_l, jnp.power(10, log_bend_freq), alpha_h], free_parameters=[True]*4)
    model_ACV = PSDToACV(psd, S_low=100,
                S_high=20,
                T=time[-1] - time[0],
                dt=np.min(np.diff(time)),
                method='SHO',
                n_components=n_component,
                estimate_variance=True,
                init_variance=jnp.var(y, ddof=1),
                use_celerite=False,
                use_legacy_celerite=False )
    
    kernel = model_ACV.ACVF
    
    mean_params = {'A1': A1, 'A2':A2, 't0':t0}
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=partial(sine_curve_model, mean_params))
    
    return gp.log_probability(y)


#One bend power law + sine generative prior
def OBPLsine_linear_generative_prior(time  = None):
    alpha_l = yield jaxns.Prior( tfpd.Uniform(-0.25, 2), name = 'alpha_l')
    alpha_h = yield jaxns.Prior( tfpd.Uniform(alpha_l , 4), name = 'alpha_h')
    # log_bend_freq = yield jaxns.Prior( tfpd.Uniform(np.log10(1/(time[-1]-time[0])),np.log10( 1/np.min(np.diff(time)))), name = 'log_bend_freq') # 1/years
    log_amp = yield jaxns.Prior( tfpd.Normal(-3, 2), name = 'log_amp')
    log_bend_freq = yield jaxns.Prior( tfpd.Normal(-2, 1), name = 'log_bend_freq')
    A1 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A1')
    A2 = yield jaxns.Prior( tfpd.Normal(0,0.3), name = 'A2')
    t0 = yield jaxns.Prior( tfpd.Uniform(0, 8), name ='t0')
    err_scale = yield jaxns.Prior(tfpd.Uniform(0.0,1.5), name='err_scale')
    m = yield jaxns.Prior(tfpd.Uniform(-2,2), name='m')
    b = yield jaxns.Prior(tfpd.Uniform(-2, 2), name='b')
    
    return alpha_l, alpha_h, log_bend_freq, log_amp, A1, A2, t0, err_scale, m, b


#One bend power law + sine likelihood
def OBPLsine_linear_log_likelihood_model(alpha_l, alpha_h, log_bend_freq, log_amp, A1, A2, t0, err_scale, m, b, n_component=10, time = None, y = None, y_errs = None ):
    
    psd = OneBendPowerLaw([jnp.power(10, log_amp), alpha_l, jnp.power(10, log_bend_freq), alpha_h], free_parameters=[True]*4)
    model_ACV = PSDToACV(psd, S_low=100,
                S_high=20,
                T=time[-1] - time[0],
                dt=np.min(np.diff(time)),
                method='SHO',
                n_components=n_component,
                estimate_variance=True,
                init_variance=jnp.var(y, ddof=1),
                use_celerite=False,
                use_legacy_celerite=False )
    
    kernel = model_ACV.ACVF
    
    mean_params = {'A1': A1, 'A2':A2, 't0':t0, 'm':m, 'b':b}
    
    gp = GaussianProcess(kernel, time, diag=(y_errs*err_scale)**2, mean=partial(sine_linear_model, mean_params))
    
    return gp.log_probability(y)




