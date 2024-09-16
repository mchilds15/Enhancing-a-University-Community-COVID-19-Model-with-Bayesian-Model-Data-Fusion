#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Created on Wed Sep 22 06:03:00 2021
@author: admin
"""


import numpy as np
import pandas as pd 
from Run_Model_Functions import run_model
from Model_Setup_And_Run_Functions import Randomly_Generated_Setup_And_Run_Model_Function, Sobol_Model_Setup_and_Run_Function, CSV_Model_Setup_And_Run_Function, VE_Benchmark_Setup_And_Run_Model_Function, Benchmark_Setup_And_Run_Model_Function
from Model_Setup_And_Run_Functions import RMSE_func
from datetime import datetime
import math 
# import matplotlib.pyplot as plt
from pyDOE import *
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform, truncnorm
from scipy.interpolate import interp1d
import joblib
import multiprocess as mp
from scipy.optimize import differential_evolution, Bounds
from RMSE_func import RMSE_Func_calc
from log_prior_functions import log_prior_pdf
from log_prior_functions import log_posterior_score
from log_prior_functions import neg_log_posterior_score
from MCMC_func import MCMC, calibration_benchmark_test_plots, history_plots_and_convergence,  log_likelihood_weights
from Param_and_Data_Organization_Funcs import VE_Benchmark_params, Benchmark_params, CSV_params,  Sobol_params, random_gen_params, calibrated_param_samples,RIT_Data_ReOrg
from progress.bar import IncrementalBar
from Data_Collection_Scripts import Sobol_Data_Collection, RMSE_Calc_Data_Collection, Random_Forest_Data_Collection
# from MCMC_func import mcmc_test
# from chainconsumer import ChainConsumer
from tqdm import tqdm, trange
import multiprocessing
        
#Start timing script
startTime = datetime.now()

CSV_startTime = datetime.now()

#Read in necessary CSVs
parameter_settings=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Model_Parameter_Inputs.csv')
parameter_settings=parameter_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
input_settings=pd.read_csv (r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Model_Settings_Inputs.csv')   #reads the csv file with the model settings values
input_settings=input_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
RIT_testing_datas=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Simplified_RIT_Data_Fall20.csv') #reads the csv file with RIT testing data
RIT_testing_data=RIT_testing_datas.replace(np.nan, 0, regex=True) #Replaces empty cells with 0 instead of nan so we can use them to find the sum between the codes
print('READ in CSVS',datetime.now() - CSV_startTime) 



#Testing Inputs 
testing_data_input=input_settings.iloc[13,1] #signals whether testing data is used for subpopulation model

#Now we import benchmark test parameters ONLY if we need them

##Global variables 
if input_settings.iloc[0,1]=='': #signifies we are running Sobol sensitivity analysis
    print('Sobol Global Variables')
    Num_params=12 # number of parameters we are running Sobol for 
    Nsim=int(input_settings.iloc[20,1]*(Num_params+2))
    N_Sobol=int(input_settings.iloc[20,1])
   
else:
    Nsim=int(input_settings.iloc[0,1]) #number of simulations
Nsub=int(input_settings.iloc[2,1]) #number of subpopulations must be at least 2 if not use other model

#Warning for N Input (number of time steps)
if (RIT_testing_data.empty == False  and input_settings.iloc[1,1] != ''): #Checks to see if either the testing data CSV or N parameter option are empty
    print('WARNING!!: Both testing data and N parameter have been provided. N parameter will be used.') #If neither are empty prints a warning in the command window to alert that both have been inputted but parameter value will be used by default

#N Input (number of time steps)
if (input_settings.iloc[1,1]) != '': # Checks to see if parameter input cell is empty or not
    print('N parameter used')
    # If it is not empty then pulls parameter from that cell to use for N (number of time steps)
    N=int(input_settings.iloc[1,1])#How often people are screened (in days)

else:
    N=np.int64((len(RIT_testing_data.iloc[:,1])*3)) #calculates N from the number of days we have data for 
    input_settings.iloc[1,1]=N #adds it to settings list

if input_settings.iloc[26,1] != '': # when a calibration is run finds out how many number of chains we want to generate the correct number of parameter sets
    Num_Chains = int(input_settings.iloc[26,1])
    Num_Params = int(input_settings.iloc[27,1])
    
# Initial compartment populations when testing data is used
if testing_data_input == 1: # Checks to see if we are using testing data
    input_settings.iloc[10,1]=round(RIT_testing_data.iloc[0,4])
    input_settings.iloc[5,1]=int(input_settings.iloc[3,1])-int(input_settings.iloc[10,1])-int(input_settings.iloc[9,1])
    
initializations_startTime = datetime.now()
#Initilizalizations  
tests_per_day=np.zeros(int((N-1)/3)) #Initializes empty vector to fill
tests_per_cycle=np.zeros(N) #Initializes empty vector to fill
simulation_number_vec=np.zeros(int(Nsim)) #Initializes empty vector to fill
RMSE_vec=np.zeros(int(Nsim)) #Initializes empty vector to fill
isolation_diff=np.zeros(int(N/3)) #Initializes empty vector to fill
isolation_pool=np.zeros(int(N/3)) #Initializes empty vector to fill
conservation_check_matrix=np.zeros((int(N), int(Nsim))) #Initializes empty matrix to fill
cycles_per_test_parameter=np.zeros((int(N), int(Nsub))) #Initializes empty matrix to fill
cycles_per_test_data=np.zeros((int(N), int(Nsub))) #Initializes empty matrix to fill
tests_per_day=np.zeros(int((N-1)/3)) #Initializes empty vector to fill
tests_per_cycle=np.zeros(int(N)) #Initializes empty vector to fill
parameters_sp=np.zeros((9,int(Nsub))) #Initializes empty matrix to fill
percent_subpop=np.zeros(Nsub) #Initializes empty matrix to fill
gamma_v=np.zeros((N, Num_Chains)) # sets up an empty vector to fill
gamma_v_end=np.zeros(Nsim) #Initializes empty vector to fill
Y=np.zeros((2,Num_Chains)) # Initializes empty matrix to fill
new_infections_per_shock_input=np.zeros(Nsub) #Initializes empty vector to fill
MoM_parameters=np.zeros((Nsim,11)) #Initializes empty matrix to fill
max_iso_population_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
max_vaxxed_iso_population_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
max_unvaxxed_iso_population_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
cumulative_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
cumulative_sympto_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
deaths_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
vaxxed_deaths_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
unvaxxed_deaths_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
vaxxed_sympto_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fillb[]\\\\\\\\\\\\\\\\\
unvaxxed_sympto_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
neg_log_posterior_score_output=np.zeros(Nsim) #Initializes empty vector to fill
#param_values_to_save=np.zeros((9,Nsim)) #Initializes empty matrix to fill
sim_block_vec=np.zeros(9*Nsim) #Initializes empty vector to fill
parameters_chain = np.zeros((Num_Params + 1, Num_Chains))
# initial_param_sets = np.zeros((Num_Params , Num_Chains))

print('Initializartions',datetime.now() - initializations_startTime) 
settings=input_settings.iloc[:,1] #selects setting values from the settings csv file

#Percent Subpopulation
for i in range(0,Nsub):
    percent_subpop[i]=float(parameter_settings.iloc[13,i+1])/100 #percent of total population in subpopulationsool;'
    
#Testing Inputs 
testing_data_input=input_settings.iloc[13,1] #signals whether testing data is used for subpopulation model
vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv



# THE UPDATED RIT POSITIVE TESTS VECTOR TO FIX THE POSITIVE RETURNS ON THE WEEKEND AND ADD THEM TO THE MONDAY 
# RETURNS SO WE DO NOT HAVE DAYS WOTH NO TESTS ADMINITSERED AND POSITIVE TESTS RETURNED
if settings[36] ==1:
    updated_positive_tests = RIT_testing_data.iloc[:,3]
    print('running model experiment')
else:
    updated_positive_tests = RIT_Data_ReOrg(RIT_testing_data)

# # below is for a normal calibration
# #The important values for the distributions
# param_dists=(( 14, 0.5), # frequency of exogenous shocks
# ( 4/11, 10, 11), # time to recovery
# ( 5/9, 5, 45), # percent advacning to system
# ( 0.12, 0.8, 1.7 ), #Rt
# ( 0.05, 0, 0.01), #symptom case fatality ratio
# ( 2/9, 3, 9), # days to incubation
# (0.5, 0.7, 0.2), # test sensitivity
# ( 1.23, 0.007), # test specificity
# (5, 0.25), # new infections per shock 
# (12, 0.5), # initial infected 
# (100, 0.2), # delay
# (-3.33, 3.33, 0.05, 0.015) # non-compliance
# )

# below is for a normal calibration
#The important values for the distributions
param_dists=(( 14, 0.5), # frequency of exogenous shocks
( 4/11, 10, 11), # time to recovery
( 5/9, 5, 45), # percent advacning to system
( 0.12, 0.8, 1.7 ), #Rt
( 0.05, 0, 0.01), #symptom case fatality ratio
( 2/9, 3, 9), # days to incubation
(0.5, 0.7, 0.2), # test sensitivity
( 1.23, 0.007), # test specificity
(5, 0.25), # new infections per shock
(12, 0.5), # initial infected 
(-1.5, 3.5, 3, 2) #inital iso

)



# The starting step size for each parameter
# # below is for normal calibration
# step_size = np.array((14/100, 11/120, 12/100, 1.7/100, 0.01/100,  9/100, 0.2/200, 0.0004, 1/100, 10/100, 70/100, 0.1/100))
# below is for normal calibration
step_size = np.array(( 14/100, 11/120, 12/100, 1.7/100, 0.01/100,  9/100, 0.2/200, 0.0004, 1/100,  10/100, 6/100))



num_iter = int(input_settings.iloc[28,1]) #number of iterations to run the Markov Chain for
rand_seed = int(input_settings.iloc[29,1]) # the random seed to reproduce a specific calibration, if no random seed this is automaticallt -1


# # below is for normal calibration
# initial_param_sets = np.array([[7, 14, 2, 10],
#                                 [11, 15, 18, 20],
#                                 [15, 23, 30, 49],
#                               [1, 0.9, 1.5, 2.4],
#                               [0.001, 0.005, 0.007, 0.01],
#                               [4, 6, 8, 11],
#                               [0.72, 0.75, 0.8, 0.89],
#                               [1.215, 1.225, 1.235, 1.245],
#                               [5, 10, 30, 60],
#                               [5, 11, 20, 40],
#                               [10, 20, 40, 60],
#                               [0.04, 0.02, 0.05, 0.08]
#                               ])

if input_settings.iloc[38,1] != -1: 
    # print('Running version')
    rand_seed_param =int(input_settings.iloc[38,1])
    np.random.seed(rand_seed_param)
    
else:
    rand_seed_param = np.random.randint(0, 1000, size = 1)
    np.random.seed(rand_seed_param)

# =============================================================================
#     ### LHS SAMPLING##
# =============================================================================
    
frequency_ES = lhs(1, samples = Num_Chains);
time_recovery = lhs(1, samples = Num_Chains);
advancing_symptoms = lhs(1, samples = Num_Chains);
Rt = lhs(1, samples = Num_Chains);
symptom_case_fatality_ratio = lhs(1, samples = Num_Chains);
days_to_incubation = lhs(1, samples = Num_Chains);
test_specificity = lhs(1, samples = Num_Chains);
test_sensitivity = lhs(1, samples = Num_Chains);
new_infections_per_shock  = lhs(1, samples = Num_Chains);
initial_infect = lhs(1, samples = Num_Chains);
initial_iso = lhs(1, samples = Num_Chains);

freq_ES_dist, time_recovery_dist, advancing_to_symptoms_dist,  Rt_dist, symptom_case_fatal_dist, days_to_incubation_dist, test_sensitivity_dist, test_specificity_dist,  new_infections_per_shock_dist, initial_infect_dist, initial_iso_dist = param_dists  


#bar = IncrementalBar('Random Params Pt 2', max = Nsim)
for i in range (Num_Chains):
    frequency_ES[i]=binom(freq_ES_dist[0], freq_ES_dist[1]).ppf(frequency_ES[i]);
    time_recovery[i]=triang(time_recovery_dist[0], loc = time_recovery_dist[1], scale = time_recovery_dist[2]).ppf(time_recovery[i]);
    advancing_symptoms[i]=triang(advancing_to_symptoms_dist[0], loc = advancing_to_symptoms_dist[1], scale = advancing_to_symptoms_dist[2]).ppf(advancing_symptoms[i]);
    Rt[i]=triang(Rt_dist[0], loc = Rt_dist[1], scale = Rt_dist[2]).ppf(Rt[i,:]);
    symptom_case_fatality_ratio[i]=triang(symptom_case_fatal_dist[0], loc = symptom_case_fatal_dist[1], scale = symptom_case_fatal_dist[2]).ppf(symptom_case_fatality_ratio[i]);
    days_to_incubation[i]=triang(days_to_incubation_dist[0], loc = days_to_incubation_dist[1], scale = days_to_incubation_dist[2]).ppf(days_to_incubation[i]);
    # test_specificity[i,:]=triang(c = 0.6, loc=0.95, scale=0.05).ppf(test_specificity_sample[i,:]);
    test_specificity[i]=norm(loc = test_specificity_dist[0], scale = test_specificity_dist[1]).ppf(test_specificity[i]);
    test_sensitivity[i]=triang(test_sensitivity_dist[0], loc = test_sensitivity_dist[1], scale = test_sensitivity_dist[2]).ppf(test_sensitivity[i]);
    new_infections_per_shock[i]=nbinom(new_infections_per_shock_dist[0], new_infections_per_shock_dist[1]).ppf(new_infections_per_shock[i]);
    initial_infect[i]=nbinom(initial_infect_dist[0], initial_infect_dist[1]).ppf(initial_infect[i]);
    initial_iso[i] = truncnorm(initial_iso_dist[0], initial_iso_dist[1], loc = initial_iso_dist[2], scale = initial_iso_dist[3]).ppf(initial_iso[i])
   
frequency_ES = np.transpose(frequency_ES)
time_recovery = np.transpose(time_recovery)
advancing_symptoms = np.transpose(advancing_symptoms)
Rt = np.transpose(Rt)
symptom_case_fatality_ratio = np.transpose(symptom_case_fatality_ratio)
days_to_incubation = np.transpose(days_to_incubation)
test_specificity = np.transpose(test_specificity)
test_sensitivity = np.transpose(test_sensitivity)
new_infections_per_shock = np.transpose(new_infections_per_shock)
initial_infect = np.transpose(initial_infect)
initial_iso = np.transpose(initial_iso)

initial_param_sets = np.vstack((frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, test_sensitivity, test_specificity, new_infections_per_shock, initial_infect, initial_iso))
# initial_param_sets = np.array([[7, 14, 2, 10], # frequency of exogenous shocks
#                                 [11, 15, 18, 20], # time to recovery
#                                 [15, 23, 30, 49], # percent advacning to system
#                                 [1, 0.9, 1.5, 2.4], #Rt
#                               [0.001, 0.005, 0.007, 0.01], #symptom case fatality ratio
#                               [4, 6, 8, 11], # days to incubation
#                               [0.72, 0.75, 0.8, 0.89], # test sensitivity
#                               [1.215, 1.225, 1.235, 1.245], # test specificity
#                               [5, 10, 30, 60],# new infections per shock
#                               [5, 11, 20, 40] # initial infected

                              # ])

# calculate weights for likelihood function
weights_norm = log_likelihood_weights(RIT_testing_data, Use_weights = False)

# checks if adaption for covariance matrix is off or on
if settings[37] == 1: # turns adapting off
    print('Adaption off')
    adaption_switch = False # turns adapting of
    covar_input = pd.read_csv('/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/covar_test[906].csv') # reads in cov matrix to use
    covar_input = covar_input.iloc[:,1:]
else:
    print('Adaption On')
    adaption_switch = True # keeps adapting on
    covar_input = None # does not provide a covariance matrix to use
    

# Other Info that the model needs
args=((settings, percent_subpop, vaccination_status, RIT_testing_data, N, Nsub, Nsim, input_settings, updated_positive_tests, weights_norm ))


inputs = [] # defines empty list to fill

# we first define a function that pacakges the arguments we need to run the chains so that the function to parrallize only needs to read in one item
for i in range(Num_Chains):
    inputs.append([num_iter, step_size, initial_param_sets[:,i], param_dists, args, rand_seed, adaption_switch, covar_input])
mcmc_outputs = {}

# Now we run the number of MCMC chains we have designated

# we now define a function that runs the chaims so that we can run them in parralllel
def run_chains(arguments):
   
    # unpacks variables we need from arguments
    num_iter = arguments[0]
    step_size = arguments[1]
    initial_param_sets= arguments[2]
    param_dists = arguments[3]
    args = arguments[4]
    rand_seed = arguments[5]
    
    # runs the calibration
    (mcmc_output, mcmc_log_output, num_sucess_output, rand_seed_output, covar_output ) = MCMC(num_iter, step_size, initial_param_sets, param_dists, args, rand_seed, adaption_switch, covar_input)
    
    # stores funciton output
    mcmc_outputs = {'mcmc_output_' + str(i) : mcmc_output,
                   'mcmc_log_output_' + str(i) : mcmc_log_output,
                   'num_success_output' + str(i) : num_sucess_output,
                   'rand_seed_outpus' + str(i) : rand_seed_output,
                   'covar matrix'+ str(i) : covar_output}    
    # returns stored function outputs
    return(mcmc_outputs)
print('running')
## runs chains in parallel and saves the results to a list
results = []
results.append(joblib.Parallel(n_jobs=4)(joblib.delayed(run_chains)(arguments) for arguments in inputs))

# for i in range(Num_Chains):
    
    
#     (mcmc_output, mcmc_log_output, num_sucess_output, rand_seed_output, covar_output ) = MCMC(num_iter, step_size, initial_param_sets[:,i], param_dists, args, rand_seed )
#     mcmc_outputs[i] = {'mcmc_output_' + str(i) : mcmc_output,
#                         'mcmc_log_output_' + str(i) : mcmc_log_output,
#                         'num_success_output' + str(i) : num_sucess_output,
#                         'rand_seed_output' + str(i) : rand_seed_output}

# for i in range(Num_Chains):
    
    
#     (mcmc_output, mcmc_log_output, num_sucess_output, rand_seed_output ) = MCMC(num_iter, step_size, initial_param_sets[:,i], param_dists , args, rand_seed )
#     mcmc_outputs[i] = {'mcmc_output_' + str(i) : mcmc_output,
#                         'mcmc_log_output_' + str(i) : mcmc_log_output,
#                         'num_success_output' + str(i) : num_sucess_output,
#                         'rand_seed_output' + str(i) : rand_seed_output}


 
if input_settings.iloc[25,1] == 1: # this signifies the calibration benchmark is being run and we return 0 so that we are sampling from priors   
    convergence_threshold = 200000
    mcmc_output_0 = mcmc_outputs[0]['mcmc_output_0']
    mcmc_output_1 = mcmc_outputs[1]['mcmc_output_1']
    calibration_benchmark_test_plots(mcmc_output_0, mcmc_output_1, convergence_threshold)

print('Total time for all simulations', datetime.now() - startTime) #prints how long the script took