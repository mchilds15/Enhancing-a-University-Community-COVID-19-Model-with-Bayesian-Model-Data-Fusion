
   

   
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
import matplotlib.pyplot as plt
from pyDOE import *
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import multiprocess as mp
from Param_and_Data_Organization_Funcs import VE_Benchmark_params, Benchmark_params, CSV_params,  Sobol_params, random_gen_params, calibrated_param_samples, testing_term_benchmark_params
from progress.bar import IncrementalBar
from Data_Collection_Scripts import Sobol_Data_Collection, RMSE_Calc_Data_Collection, Random_Forest_Data_Collection, Data_storage_paper_one_figures,  Metrics_of_Interest_Collection_Subpops, Metrics_of_Interest_Collection_Onepop, Testing_Data_Collection
from tqdm import tqdm, trange

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

Benchmark_startTime = datetime.now()

#Testing Inputs 
testing_data_input=input_settings.iloc[13,1] #signals whether testing data is used for subpopulation model

#Now we import benchmark test parameters ONLY if we need them
if input_settings.iloc[14,1]==1:
    print('Using Benchmark')
    benchmark_parameter_settings=pd.read_csv(r'/Users/meghanchilds/Desktop/CSVs to run Model/Benchmark CSVs/Model_Parameter_Inputs_Benchmark.csv')
    benchmark_parameter_settings=benchmark_parameter_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code

if input_settings.iloc[15,1]==1:
    print('Using VE Benchmark')
    VE_Benchmark_parameter_settings=pd.read_csv(r'/Users/meghanchilds/Desktop/CSVs to run Model/Benchmark CSVs/VE_Benchmark_Parameters.csv')
    VE_Benchmark_parameter_settings=VE_Benchmark_parameter_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
print('Check for bencmarks',datetime.now() - Benchmark_startTime) 

if input_settings.iloc[33,1]==1:
    print('Using VE Benchmark')
    Testing_term_Benchmark_parameter_settings = pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Benchmark Testing/Model_Parameter_Inputs_New_Testing_Benchmark.csv')
    Testing_term_Benchmark_parameter_settings = Testing_term_Benchmark_parameter_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
print('Check for bencmarks',datetime.now() - Benchmark_startTime) 

##Global variables
if input_settings.iloc[0,1]=='': #signifies we are running Sobol sensitivity analysis
    print('Sobol Global Variables')
    Num_params=12 # number of parameters we are running Sobol for 
    Nsim=int(input_settings.iloc[20,1]*(Num_params+2))
    N_Sobol=int(input_settings.iloc[20,1])
    
elif input_settings.iloc[22,1] == 1: #signifies the data is being run with the calibrated samples
    
    calibrated_param_samples_csv=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Calibrated Parameters/Fall_Samples_For_Analysis_FLF_1000_sims_new_std_weight_LL_multiply_4_sep_24.csv')
    Nsim=int(len(calibrated_param_samples_csv.iloc[:,1])) # if calibrated samples are being used it sets number of simulations to number of samples provided

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
    N = np.int64((len(RIT_testing_data.iloc[:,1])*3))  #calculates N from the number of days we have data for 
    input_settings.iloc[1,1]=N #adds it to settings list
    

# Initial compartment populations when testing data is used
if testing_data_input == 1: # Checks to see if we are using testing data
    input_settings.iloc[10,1] = round(RIT_testing_data.iloc[0,4])
    input_settings.iloc[5,1] = int(input_settings.iloc[3,1]) - int(input_settings.iloc[10,1])- int(input_settings.iloc[9,1])
    
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
# cycles_per_test_data=np.zeros((int(N), int(Nsub))) #Initializes empty matrix to fill
tests_per_day=np.zeros(int((N-1)/3)) #Initializes empty vector to fill
tests_per_cycle=np.zeros(int(N)) #Initializes empty vector to fill
parameters_sp=np.zeros((9,int(Nsub))) #Initializes empty matrix to fill
percent_subpop=np.zeros(Nsub) #Initializes empty matrix to fill
gamma_v=np.zeros((N)) # sets up an empty vector to fill
gamma_v_end=np.zeros(Nsim) #Initializes empty vector to fill
Y=np.zeros((2,Nsim)) # Initializes empty matrix to fill
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
vaxxed_sympto_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
unvaxxed_sympto_infections_all_sims=np.zeros(Nsim) #Initializes empty vector to fill
max_total_new_infect_2_days_all_sims = np.zeros(Nsim)
max_vaxxed_new_infect_2_days_all_sims = np.zeros(Nsim)
max_unvaxxed_new_infect_2_days_all_sims = np.zeros(Nsim)
max_total_new_infect_week_all_sims = np.zeros(Nsim)
max_vaxxed_new_infect_week_all_sims = np.zeros(Nsim)
max_unvaxxed_new_infect_week_all_sims = np.zeros(Nsim)

print('Initializations',datetime.now() - initializations_startTime) 

#Percent Subpopulation
for i in range(0,Nsub):
    percent_subpop[i]=float(parameter_settings.iloc[13,i+1])/100 #percent of total population in subpopulationsool;'

    
#Uses Pararmeters inPutted in CSV file for VE benchmark test#
if input_settings.iloc[15,1]==1:
    
    print('Using parameters from VE Benchmark CSV')
    
    (frequency_ES, time_recovery, advancing_symptoms, 
     Rt, symptom_case_fatality_ratio, days_to_incubation, 
     time_to_return_fps, test_specificity, test_sensitivity,  
     gamma_v_start, gamma_u, 
     new_infections_per_shock_og) = VE_Benchmark_params(N, Nsub, Nsim, 
                                                        VE_Benchmark_parameter_settings)

##Uses Pararmeters inPutted in CSV file for benchmark test#
elif input_settings.iloc[14,1]==1:
    
    print('Using parameters from Benchmark CSV')
    
    (frequency_ES, time_recovery, advancing_symptoms, 
     Rt, symptom_case_fatality_ratio, days_to_incubation, 
     time_to_return_fps, test_specificity, test_sensitivity, 
     gamma_v_start, gamma_u, 
     new_infections_per_shock_og) = Benchmark_params(N, Nsub, Nsim, 
                                                     benchmark_parameter_settings)
                                                     
 ##Uses Pararmeters inPutted in CSV file for testing term benchmark test#
elif input_settings.iloc[33,1]==1:
          
   print('Using parameters from Testing term Benchmark CSV')
          
   (frequency_ES, time_recovery, advancing_symptoms, 
    Rt, symptom_case_fatality_ratio, days_to_incubation, 
    time_to_return_fps, test_specificity, test_sensitivity, 
    gamma_v_start, gamma_u, 
    new_infections_per_shock_og, delay, non_compliance) = testing_term_benchmark_params(N, Nsub, Nsim, 
                                                           Testing_term_Benchmark_parameter_settings)                                               
    
##Uses Pararmeters inPutted in CSV file #
elif parameter_settings.iloc[0,1] != '':
    
     print('Using parameters from CSV')
     
     (frequency_ES, time_recovery, advancing_symptoms, 
      Rt, symptom_case_fatality_ratio, days_to_incubation, 
      time_to_return_fps, test_specificity, test_sensitivity,  
      gamma_v_start, gamma_u, 
      new_infections_per_shock_og, delay, non_compliance) = CSV_params(N, Nsub, Nsim, 
                                                parameter_settings)


elif input_settings.iloc[19,1]==1:
    
    print('Generating Sobol Parameters')
    
    (param_values_test, gamma_v_start, new_infections_per_shock_og,
     gamma_v_mod, time_to_return_fps, gamma_u) = Sobol_params(N, Nsub, 
                                 Nsim, N_Sobol, 
                                 Num_params)
                                 
elif input_settings.iloc[22,1] == 1:
    
    print('Using Calibrated Samples')

    # First provide fixed param values
    # freq_ES_fix = 7
    # symptom_case_fatality_ratio_fix = 0.0005
    # new_infections_per_shock_fix = 2
    
    (frequency_ES, time_recovery, advancing_symptoms, 
     Rt, symptom_case_fatality_ratio, days_to_incubation, 
     time_to_return_fps, test_specificity, test_sensitivity,  
     gamma_v_start, gamma_v_mod,  gamma_u, 
     new_infections_per_shock_og, initial_infect, delay, non_compliance, initial_iso) = calibrated_param_samples(N, Nsub, input_settings, calibrated_param_samples_csv)
    
else:
    LHS_startTime = datetime.now()
    
    #Generate Matrix of Randomly Generated Parameter sets using latin hypercube sampling## 
    
    print('Generating random parameters')
    
    #Latin Hypercube Sampling
    
    (frequency_of_screening, frequency_ES, time_recovery, 
     advancing_symptoms, Rt, symptom_case_fatality_ratio, 
     days_to_incubation, time_to_return_fps, test_specificity, 
     test_sensitivity,  gamma_v_start, gamma_v_mod,  gamma_u, 
     new_infections_per_shock_og, initial_infect, delay, non_compliance) = random_gen_params(N, Nsub, Nsim, input_settings)
   
    print('LHS',datetime.now() - LHS_startTime)    
    

    
    
if input_settings.iloc[14,1]==1: # is triggered when one of the benchmark tests is being run
    
    print('VE for OG Benchmark Tests')    # for deubbing
    
    for i in range(0,Nsim):
        gamma_v_end[i]=gamma_v_start[0] # sets vaccine efficacy so it does not decrease
        
        Y[0,i]=gamma_v_start[0]
        Y[1,i]=gamma_v_end[0]
        
elif input_settings.iloc[15,1]==1: # is triggered when one of the benchmark tests is being run
   
    print('VE for VE Benchmark Tests')    # for deubbing
    
    for i in range(0,Nsim):
        gamma_v_end[i]=gamma_v_start # sets vaccine efficacy so it does not decrease

        Y[0,i]=gamma_v_start
        Y[1,i]=gamma_v_end[i]

#Y=np.array([float(gamma_v_start), float(gamma_v_end)])
elif input_settings.iloc[16,1]==0: # checks if switch for decreasing efficacy is off
       
        print('Steady VE') # for deubbing

        gamma_v_end=np.zeros(len(gamma_v_start))

        for i in range (0,Nsim):
            gamma_v_end[i]=gamma_v_start[i] #sets gamma start to gamma end when we do not have decreasing efficacy
        
            Y[0,i]=gamma_v_start[i]
            Y[1,i]=gamma_v_end[i]

elif input_settings.iloc[16,1]==1 and input_settings.iloc[19,1]==1:  # checks if switch for decreasing efficacy is on and Sobol is being run
       
       print('Sobol and Decreasing VE')

       for i in range(0,Nsim): 
           gamma_v_end[i]=gamma_v_start[i]+(1-gamma_v_start[i])*gamma_v_mod[i] # defines gamma_v_end to be the value of gamma_v_start modulated by our gamma_v_mod parameter
           
           Y[0,i]=gamma_v_start[i] # loads gamma_v_start values into the first column of Y to be used for interpolating the line between gamma_v_start and gamma_v_end
           Y[1,i]=gamma_v_end[i] # loads gamma_v_end value into 2nd column of Y to be used for interpolating the line between gamma_v_start and gamma_v_end

else: # if switch for decreasing efficacy is on we set it up to run with decreasing efficacy

    print('Decreasing VE') # for deubbing   

    for i in range (0,Nsim):
        gamma_v_end[i]=gamma_v_start[i]+(1-gamma_v_start[i])*gamma_v_mod[i] # defines gamma_v_end to be the value of gamma_v_start modulated by our gamma_v_mod parameter
       
        Y[0,i]=gamma_v_start[i]  # loads gamma_v_start values into the first column of Y to be used for interpolating the line between gamma_v_start and gamma_v_end
        Y[1,i]=gamma_v_end[i] # loads gamma_v_end value into 2nd column of Y to be used for interpolating the line between gamma_v_start and gamma_v_end

X=np.array([0, N-1]) #sets our X values for our interpolation as our first and last indices 
        

#Testing Data Case
if testing_data_input == 1: # Checks to see if we are using testing data
    
    print('Running Testing Data Case')
    for i in trange(0,Nsim): 
        # print('Testing data used in subpop model')
        # testing_startTime = datetime.now()
        # for k in range(0,int(((N-1)/3))):
        #     tests_per_day[k]=RIT_testing_data.iloc[k,2] #Imports data for tests per day from the csv
        # for j in range(0,int(((N-1)/3))):
        #     tests_per_cycle[3*j:3*j+3]=tests_per_day[j]/3 #converts daily a data to tests per time step
        # for h in range(0,N):
        #     if tests_per_cycle[h]==0: # if no tests have occured on this day automatically set cycles_per_test to zero to avoid divide by zero error
        #         cycles_per_test_data[h,:]=0
        #     else:
        #         cycles_per_test_data[h,:]=(input_settings.iloc[3,1])/(tests_per_cycle[h]) #Uses testing data to calculate cycles per test
       
        # print('testing',datetime.now() - testing_startTime) 
        
        if Nsub > 1 and input_settings.iloc[15,1] != 1:
            if i == 0:
                print('Splitting exogenous shocks proportionally') # for debugging
            for n in range(0,Nsub):
                new_infections_per_shock_input[n]=new_infections_per_shock_og[i]*percent_subpop[n] #splits up exogenous shocks proportionally
        else:
            if i == 0:
                print('NOT splitting exogenous shocks proportionally') # for debugging
            new_infections_per_shock_input = np.array(new_infections_per_shock_og[i])
            new_infections_per_shock_input  = new_infections_per_shock_input.reshape((1,1))
        
        parameters_startTime = datetime.now()  
        
        #Combines paramaters into a matrix and gives new_infections per shock its own vector/matrix (depends on how many subpopulations we have)
        if input_settings.iloc[33,1] == 1:
            print('Testing term benchmark params')
            initial_infect = input_settings.iloc[9,1] # just as a place holder to keep same places in vector assigned]
            parameters=np.vstack((frequency_ES, time_recovery, advancing_symptoms, 
                                  Rt, symptom_case_fatality_ratio, days_to_incubation, 
                                  time_to_return_fps, test_sensitivity, test_specificity, 
                                  gamma_u, initial_infect, delay, non_compliance))
            
            new_infections_per_ES_input=new_infections_per_shock_input
        
        elif parameter_settings.iloc[0,1] != '': #if we use parameters all parameter values are combined together into a matrix
            print('CSV Parameters')    
            initial_infect = input_settings.iloc[9,1] # just as a place holder to keep same places in vector assigned]
            parameters=np.vstack((frequency_ES, time_recovery, advancing_symptoms, 
                                  Rt, symptom_case_fatality_ratio, days_to_incubation, 
                                  time_to_return_fps, test_sensitivity, test_specificity, 
                                  gamma_u, initial_infect, delay, non_compliance))
            
            new_infections_per_ES_input=new_infections_per_shock_input
            
        elif input_settings.iloc[19,1]==1:
            
            print('Random Parameters for Sobol')
            
            parameters=np.zeros((10, Nsub))
            
            for l in range(0,6):
                    parameters[l,:]=param_values_test[i,l]
                    
            parameters[6,:]=time_to_return_fps[1,:]
            
            for l in range(7,9):        
                parameters[l,:]=param_values_test[i,l-1]
                
            parameters[9,:]=gamma_u[0,:]
            
            new_infections_per_ES_input=new_infections_per_shock_input
            
        else: #if we randomly generate parameters we pull a row of each parameter to combine into our parameter matrix for each simulation 
            if i ==0:   
                print( 'Random Parameters or Calibrated Params')
            
            if input_settings.iloc[22,1] == 1: # indicates calibrated samples are used
                if i == 0:
                    print('Calibrated Param Option')
            else: 
                if i == 0:
                    print('Randomly Generated Samples')
                
            parameters=np.vstack((frequency_ES[i,:], (time_recovery[i,:]), (advancing_symptoms[i,:]), 
                                  (Rt[i,:]), (symptom_case_fatality_ratio[i,:]), (days_to_incubation[i,:]), 
                                  (time_to_return_fps[i,:]), (test_sensitivity[i,:]), (test_specificity[i,:]),
                                  (gamma_u[i,:]), initial_infect[i,:], delay[i,:], non_compliance[i,:],new_infections_per_shock_input, initial_iso[i,:])) 
            
            new_infections_per_ES_input=new_infections_per_shock_input
            
        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
        
        vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
        
        # print('storing parameters',datetime.now() - parameters_startTime) 
        
        # testing_data=cycles_per_test_data #if we not are we set the testing data to be the one calculated using a parameter (either CSV input or randomly generated)
        
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
        
        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
        
        
        model_startTime = datetime.now()
        if Nsub > 1:
            (num_pos_tests_per_day, pos_test_results, num_pos_test, max_isolation_pop, max_unvaxxed_isolation_pop, 
                    max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                    cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                    cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                    cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                    new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                    isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, 
                                                                                      percent_subpop, new_infections_per_ES_input, 
                                                                                      vaccination_status, gamma_v, RIT_testing_data, Nsim)#run our model function
        elif Nsub ==1 and input_settings.iloc[33,1] == 1: # for testing term benchmark
            (total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference, I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
                   cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings,  
                                                                                     percent_subpop, new_infections_per_ES_input, 
                                                                                     vaccination_status, gamma_v, RIT_testing_data, Nsim)

        elif Nsub ==1 and input_settings.iloc[18,1] == 1:
            ( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, 
                                                                                     percent_subpop, new_infections_per_ES_input, 
                                                                                     vaccination_status, gamma_v, RIT_testing_data, Nsim)
        else:
              (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
              isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings, 
                                                                                   percent_subpop, new_infections_per_ES_input, 
                                                                                   vaccination_status, gamma_v, RIT_testing_data, Nsim)
       
        # print('Model Run',datetime.now() - model_startTime) 
        
        # print('Sim', i, 'complete')
        
        # simulation_number_vec_timestep=i*np.ones(N)
        # populations_RMSE=np.column_stack((simulation_number_vec_timestep, mat_subpop_totals))
        # df_populations_RMSE=pd.DataFrame(populations_RMSE, columns=['Simulation Number','Uninfected', 'False Positive', 'Exposed', 'Asymptomatic', 'Symptomatic', 'True Positive', 'Recovered', 'Deceased'])
        # df_populations_RMSE.to_csv('Population_RMSE.csv', index=False, mode='a')
       
        ##METHOD OF MORRIS AND SOBOL DATA COLLECTION
        if input_settings.iloc[17,1]==1: #signals we are collecting data for method of morris so triggers the following, this way we are not using time to run it when we do not need it
            # Storing Parameters for MoM
            MoM_parameters[i,:]=[frequency_ES[i,1], time_recovery[i,1], advancing_symptoms[i,1], Rt[i,1], symptom_case_fatality_ratio[i,1], days_to_incubation[i,1], test_sensitivity[i,1], test_specificity[i,1], gamma_v_start[i], new_infections_per_shock_og[i], frequency_of_screening[i,1]] #collects parameter values for each simulation in the form we need to run method of morris
            #Storing outputs of interest for MoM and other analysis
       
        if input_settings.iloc[17,1]==1 or input_settings.iloc[18,1]==1 or input_settings.iloc[19,1]==1 :
            if Nsub > 1:
                print('Success')
                max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
                max_vaxxed_iso_population_all_sims[i]=max_vaxxed_isolation_pop # Stores max vaxxed isolation population for each simulation
                max_unvaxxed_iso_population_all_sims[i]=max_unvaxxed_isolation_pop # Stores max unvaxxed isolation population for each simulation
                cumulative_infections_all_sims[i]=cumulative_infect_finalR # Stores cumulative infections for reach simulation
                cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulation
                vaxxed_sympto_infections=np.cumsum(symptomatic_infection_tally[:,0])
                unvaxxed_sympto_infections=np.cumsum(symptomatic_infection_tally[:,1])
                vaxxed_sympto_infections_all_sims[i]=vaxxed_sympto_infections[N-1]
                unvaxxed_sympto_infections_all_sims[i]=unvaxxed_sympto_infections[N-1]
                deaths_all_sims[i]=mat_subpop_totals[N-1,7] # Stores total deaths for each simulation
                vaxxed_deaths_all_sims[i]=mat_subpop[N-1,7,0] #Stores total vaxxed deaths for each simulation
                unvaxxed_deaths_all_sims[i]=mat_subpop[N-1,7,1] #Stores total unvaxxed deaths for each simulation
            else:
                max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
                cumulative_infections_all_sims[i]=cumulative_infect_finalR # Stores cumulative infections for reach simulation
                cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulatio
                deaths_all_sims[i]=mat_subpop_totals[N-1,7] # Stores total deaths for each simulation
                
     

            #Data Comparison
        if testing_data_input == 1: #only does calculation if we are using testing data
                for k in range(0, int(N/3)):
                    isolation_pool[k] =np.mean((isolation_pool_population[3*k], isolation_pool_population[(3*k)+1], isolation_pool_population[(3*k)+2])) #totals complete isolation pool for each day    
                    #print('run',k)
                    isolation_diff[k]=abs(RIT_testing_data.iloc[k,4]-isolation_pool[k]) #Calculates difference between isolation pools for each day
                                
                max_iso_diff=max(isolation_diff) #finds max difference between isolation pools over the semester
                min_iso_diff=min(isolation_diff) #finds min difference between isolation pools over the semester
                
                #Now we calculate our Mean Squared Error and Root Mean Squared Error and make sure its ready to be put in the dataframe
                MSE=(np.square(np.subtract(isolation_pool, RIT_testing_data.iloc[:,4]))).mean() #calculates mean squared error
                RMSE=math.sqrt(MSE) #calculates root mean sqaured error
                simulation_number_vec[i]=i #keeps track of simulation number so we know what RMSE pertains to what run number
                RMSE_vec[i]=RMSE #stores all the RMSE in a vector


# OG Benchmark
elif input_settings.iloc[14,1]==1: # when benchmark test is being run
    
    print('Running Benchmark Case')

    for i in range(0,Nsim): 

       (total_pop_difference, cumulative_infections_difference, isolation_pool_difference, 
        I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
        cumulative_infect_finalR, conservation_check, isolation_pool_population, 
        mat_subpop, mat_subpop_totals) = Benchmark_Setup_And_Run_Model_Function(i, Nsim, N, Nsub, 
                                                                                benchmark_parameter_settings,
                                                                                input_settings,frequency_ES, 
                                                                                time_recovery, advancing_symptoms,  
                                                                                Rt, symptom_case_fatality_ratio,
                                                                                days_to_incubation, time_to_return_fps, 
                                                                                test_sensitivity, test_specificity, 
                                                                                gamma_u, new_infections_per_shock_input,new_infections_per_shock_og, 
                                                                                cycles_per_test_parameter, X, Y, gamma_v, percent_subpop, RIT_testing_data)

#VE Benchmark
elif input_settings.iloc[15,1]==1: # when VE bencmark is being run
     print('Running VE Benchmark Case')
     for i in range(0,Nsim): 
        (total_pop_difference_VE,cumulative_infections_difference_VE, isolation_pool_difference_VE,
         total_pop_benchmark_VE, total_pop_difference_VE, I, 
         IDivisor, new_infections_per_shock, parameters, infection_tally, 
         cumulative_infect_finalR, conservation_check,isolation_pool_population, 
         mat_subpop, mat_subpop_totals)= VE_Benchmark_Setup_And_Run_Model_Function( i, Nsim, N, Nsub,  
                                                                                   VE_Benchmark_parameter_settings, parameter_settings,
                                                                                   input_settings, percent_subpop, 
                                                                                   new_infections_per_shock_input,new_infections_per_shock_og, 
                                                                                   frequency_ES, time_recovery, 
                                                                                   advancing_symptoms, Rt, 
                                                                                   symptom_case_fatality_ratio, 
                                                                                   days_to_incubation, time_to_return_fps, 
                                                                                   test_sensitivity,test_specificity, 
                                                                                   gamma_u, cycles_per_test_parameter, X, Y, gamma_v, RIT_testing_data)
        
#CSV file case
elif  parameter_settings.iloc[0,1] != '':#checks to see if CSV inputd are used or if the parameter is randomly generated
    
    print('Running CSV file case')
    if Nsub > 1 and input_settings.iloc[18,1] == 1 :  
        # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
                                                                        
        for i in range(0,Nsim):
            
            (num_pos_tests_per_day, pos_test_results, total_new_infections_per_2_days, total_new_infections_per_week, 
                   vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
                   unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
                   max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
                   max_unvaxxed_new_infect_week, max_isolation_pop, max_unvaxxed_isolation_pop, 
                   max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals) = CSV_Model_Setup_And_Run_Function(i, N, Nsub, 
                                                                                                                parameter_settings, 
                                                                                                                input_settings,cycles_per_test_parameter, 
                                                                                                                new_infections_per_shock_input, new_infections_per_shock_og, 
                                                                                                                percent_subpop,frequency_ES, time_recovery, advancing_symptoms, 
                                                                                                                Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, 
                                                                                                                test_sensitivity, test_specificity, gamma_u, gamma_v, X, Y, RIT_testing_data, Nsim)                                                                                                                                                                    
    elif Nsub == 1 and input_settings.iloc[18,1] == 1: 
      # This is the case where we have one subpop and we want to collect our metrics of interest
      
          (num_pos_tests_per_day, pos_test_results, total_new_infections_per_2_days, total_new_infections_per_week, 
             max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
             cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
             cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
             cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
             new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
             isolation_pool_population, mat_subpop, mat_subpop_totals) = CSV_Model_Setup_And_Run_Function(i, N, Nsub, 
                                                                                                          parameter_settings, 
                                                                                                          input_settings,cycles_per_test_parameter, 
                                                                                                          new_infections_per_shock_input, new_infections_per_shock_og, 
                                                                                                          percent_subpop, frequency_ES, time_recovery, 
                                                                                                          advancing_symptoms, Rt, symptom_case_fatality_ratio, 
                                                                                                          days_to_incubation, time_to_return_fps, test_sensitivity, 
                                                                                                          test_specificity, gamma_u, gamma_v, X, Y, RIT_testing_data, Nsim)     
                                                                                                          
    else:
          
          # This is our baseline mode where we just output base outputs
          (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
           conservation_check,isolation_pool_population, mat_subpop, mat_subpop_totals)= CSV_Model_Setup_And_Run_Function(i, N, Nsub, 
                                                                                                              parameter_settings, 
                                                                                                              input_settings,cycles_per_test_parameter, 
                                                                                                              new_infections_per_shock_input, new_infections_per_shock_og, 
                                                                                                              percent_subpop, frequency_ES, time_recovery, 
                                                                                                              advancing_symptoms, Rt, symptom_case_fatality_ratio, 
                                                                                                              days_to_incubation, time_to_return_fps, test_sensitivity, 
                                                                                                              test_specificity, gamma_u, gamma_v, X, Y, RIT_testing_data, Nsim)     
# Sobol Parameters Case
elif  input_settings.iloc[19,1] == 1:#checks to see if switch for Sobol is on
    
    frequency_of_screening_fill=np.zeros((Nsim, Nsub))

    print('Running Sobol Case')

    if input_settings.iloc[21, 1] == 1: # checks if we are running parallel if
         num_cores=mp.cpu_count()
         (results)=Parallel(n_jobs=num_cores)(delayed(Sobol_Model_Setup_and_Run_Function)(i, Nsim, N, Nsub, input_settings, parameter_settings, param_values_test, percent_subpop, time_to_return_fps, frequency_of_screening_fill, cycles_per_test_parameter, 
                                                       new_infections_per_shock_og, new_infections_per_shock_input, gamma_u, gamma_v, X, Y, max_iso_population_all_sims, 
                                                       max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                                       vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims, RIT_testing_data) for i in range(0, Nsim))
    
        
    elif input_settings.iloc[18,1]==1 : # checks to see if we are not running in paralell if we are still collecting outputs for Sobol
         
             
         if Nsub  > 1:     
                 bar = IncrementalBar('Sobol Model Runs Nsubs', max = Nsim)
                 
                 for i in range(0,Nsim): 
                 
                     ( total_new_infections_per_2_days, total_new_infections_per_week, 
                            vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
                            unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
                            max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
                            max_unvaxxed_new_infect_week, max_isolation_pop, max_unvaxxed_isolation_pop, 
                            max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                            cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                            cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                            cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                            new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                            isolation_pool_population, mat_subpop, mat_subpop_totals) = Sobol_Model_Setup_and_Run_Function(i, Nsim, N, Nsub, 
                                                                                                                                         input_settings, parameter_settings, 
                                                                                                                                         param_values_test, percent_subpop, 
                                                                                                                                         time_to_return_fps, frequency_of_screening_fill, 
                                                                                                                                         cycles_per_test_parameter, new_infections_per_shock_og, 
                                                                                                                                         new_infections_per_shock_input, gamma_u, gamma_v, X, Y, 
                                                                                                                                         max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, 
                                                                                                                                         max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, 
                                                                                                                                         cumulative_sympto_infections_all_sims,vaxxed_sympto_infections_all_sims, 
                                                                                                                                         unvaxxed_sympto_infections_all_sims, deaths_all_sims, vaxxed_deaths_all_sims, 
                                                                                                                                         unvaxxed_deaths_all_sims, RIT_testing_data)
                                                                                                                                         
                                                 
                 bar.next() 
         else:
             bar = IncrementalBar('Sobol Model Runs', max = Nsim)
             
             for i in range(0,Nsim): 
                   ( total_new_infections_per_2_days, total_new_infections_per_week, 
                          max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                          cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                          cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                          cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                          new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                          isolation_pool_population, mat_subpop, mat_subpop_totals)= Sobol_Model_Setup_and_Run_Function(i, Nsim, N, Nsub, 
                                                                                                                                 input_settings, parameter_settings, 
                                                                                                                                 param_values_test, percent_subpop, 
                                                                                                                                 time_to_return_fps, frequency_of_screening_fill, 
                                                                                                                                 cycles_per_test_parameter, new_infections_per_shock_og, 
                                                                                                                                 new_infections_per_shock_input, gamma_u, gamma_v, X, Y, 
                                                                                                                                 max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, 
                                                                                                                                 max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, 
                                                                                                                                 cumulative_sympto_infections_all_sims,vaxxed_sympto_infections_all_sims, 
                                                                                                                                 unvaxxed_sympto_infections_all_sims, deaths_all_sims, vaxxed_deaths_all_sims, 
                                                                                                                                 unvaxxed_deaths_all_sims, RIT_testing_data)
                                                                                                                               
                   
                   bar.next()                                                                                          
    else:
         bar = IncrementalBar('Sobol Model Runs', max = Nsim)
         for i in range(0,Nsim):
             
             (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
                    conservation_check,isolation_pool_population, mat_subpop, mat_subpop_totals) = Sobol_Model_Setup_and_Run_Function(i, Nsim, N, Nsub, 
                                                                                                                                 input_settings, parameter_settings, 
                                                                                                                                 param_values_test, percent_subpop, 
                                                                                                                                 time_to_return_fps, frequency_of_screening_fill, 
                                                                                                                                 cycles_per_test_parameter, new_infections_per_shock_og, 
                                                                                                                                 new_infections_per_shock_input, gamma_u, gamma_v, X, Y, 
                                                                                                                                 max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, 
                                                                                                                                 max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, 
                                                                                                                                 cumulative_sympto_infections_all_sims,vaxxed_sympto_infections_all_sims, 
                                                                                                                                 unvaxxed_sympto_infections_all_sims, deaths_all_sims, vaxxed_deaths_all_sims, 
                                                                                                                                 unvaxxed_deaths_all_sims, RIT_testing_data)
             bar.next()                                                                                                                                             
else:                                                               
    
    print('Running Randomly Generated Case')
    
    if input_settings.iloc[21,1] == 1: # checks to see if we are running parallel
        
        num_cores=mp.cpu_count() # sets number of cores
        testing_startTime = datetime.now() 
        
        if parameter_settings.iloc[1, 1]!= '': # checks to see if frequency of screening is fixed, ie a value in the CSV or if it is randomly sampled
            
            frequency_of_screening_subpop_input=np.array(parameter_settings.iloc[1,1:Nsub+1]) #How often people are screened (in days)
            
            frequency_of_screening=np.zeros((Nsim, Nsub))
           
            for m in range(0, len(parameter_settings.iloc[1,1:Nsub+1])):
                frequency_of_screening[:,m]=np.float(frequency_of_screening_subpop_input[m])
        else:
            frequency_of_screening = frequency_of_screening            
            
        bar = IncrementalBar('Randomly Gen Model Runs', max = Nsim)
        (results)=Parallel(n_jobs=num_cores)(delayed(Randomly_Generated_Setup_And_Run_Model_Function)(i, N, Nsim, Nsub, 
                                                                                                      input_settings, parameter_settings, 
                                                                                                      cycles_per_test_parameter, new_infections_per_shock_input, 
                                                                                                      new_infections_per_shock_og, percent_subpop,frequency_of_screening,
                                                                                                      frequency_ES, time_recovery, advancing_symptoms, Rt, 
                                                                                                      symptom_case_fatality_ratio, days_to_incubation, 
                                                                                                      time_to_return_fps, test_sensitivity, test_specificity, 
                                                                                                      gamma_u, gamma_v, initial_infect, X, Y, max_iso_population_all_sims, 
                                                                                                      max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims,
                                                                                                      cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                                                                                      vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, 
                                                                                                      deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims,  
                                                                                                      max_total_new_infect_2_days_all_sims,max_vaxxed_new_infect_2_days_all_sims,
                                                                                                      max_unvaxxed_new_infect_2_days_all_sims, max_total_new_infect_week_all_sims,
                                                                                                      max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, RIT_testing_data) for i in range(0, Nsim))
        bar.next()

    elif input_settings.iloc[18,1]==1 or input_settings.iloc[19,1]==1 : # checks to see if we are not running in paralell if we are still collecting outputs 
        
        #bar = IncrementalBar('Randomly Gen Model Runs', max = Nsim)
        
        for i in range (0, Nsim):
            testing_startTime = datetime.now() 
            if parameter_settings.iloc[1, 1]!= '':
                frequency_of_screening_subpop_input=np.array(parameter_settings.iloc[1,1:Nsub+1]) #How often people are screened (in days)
                frequency_of_screening=np.zeros((Nsim, Nsub))
                for m in range(0, len(parameter_settings.iloc[1,1:Nsub+1])):
                    frequency_of_screening[:,m]=float(frequency_of_screening_subpop_input[m])
            else:
                frequency_of_screening= frequency_of_screening 
        if Nsub > 1:        
            for i in trange(0, Nsim):
               #print('1st here')
               ( total_new_infections_per_2_days, total_new_infections_per_week, 
                 vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
                 unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
                 max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
                 max_unvaxxed_new_infect_week, max_isolation_pop, max_unvaxxed_isolation_pop, 
                 max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                 cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                 cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                 cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                 new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                 isolation_pool_population, mat_subpop, mat_subpop_totals, max_iso_population_all_sims, max_vaxxed_iso_population_all_sims,
                 max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, cumulative_sympto_infections_all_sims, 
                 vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims)   = Randomly_Generated_Setup_And_Run_Model_Function(i, N, Nsim, Nsub, 
                                                                                                                                             input_settings, parameter_settings, 
                                                                                                                                             cycles_per_test_parameter, new_infections_per_shock_input, 
                                                                                                                                             new_infections_per_shock_og, percent_subpop,frequency_of_screening,
                                                                                                                                             frequency_ES, time_recovery, advancing_symptoms, Rt, 
                                                                                                                                             symptom_case_fatality_ratio, days_to_incubation, 
                                                                                                                                             time_to_return_fps, test_sensitivity, test_specificity, 
                                                                                                                                             gamma_u, gamma_v, initial_infect, X, Y, max_iso_population_all_sims, 
                                                                                                                                             max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims,
                                                                                                                                             cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                                                                                                                             vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, 
                                                                                                                                             deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims,  
                                                                                                                                             max_total_new_infect_2_days_all_sims,max_vaxxed_new_infect_2_days_all_sims,
                                                                                                                                             max_unvaxxed_new_infect_2_days_all_sims, max_total_new_infect_week_all_sims,
                                                                                                                                             max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, RIT_testing_data)
                                                                                      
               #bar.next()  
        
        else: 
            for i in range(0, Nsim):
                ( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop, cumulative_symptomatic_infect_final_R, 
                   cumulative_asymptomatic_infect_final_R, cumulative_asymptomatic_infections,symptomatic_infection_tally, 
                   symptomatic_infection_tally_combined, cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined,
                   I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals, max_iso_population_all_sims, cumulative_infections_all_sims, 
                   cumulative_sympto_infections_all_sims) = Randomly_Generated_Setup_And_Run_Model_Function(i, N, Nsim, Nsub, 
                                                                                                               input_settings, parameter_settings, 
                                                                                                               cycles_per_test_parameter, new_infections_per_shock_input, 
                                                                                                               new_infections_per_shock_og, percent_subpop,frequency_of_screening,
                                                                                                               frequency_ES, time_recovery, advancing_symptoms, Rt, 
                                                                                                               symptom_case_fatality_ratio, days_to_incubation, 
                                                                                                               time_to_return_fps, test_sensitivity, test_specificity, 
                                                                                                               gamma_u, gamma_v, initial_infect, X, Y, max_iso_population_all_sims, 
                                                                                                               max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims,
                                                                                                               cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                                                                                               vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, 
                                                                                                               deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims,  
                                                                                                               max_total_new_infect_2_days_all_sims,max_vaxxed_new_infect_2_days_all_sims,
                                                                                                               max_unvaxxed_new_infect_2_days_all_sims, max_total_new_infect_week_all_sims,
                                                                                                               max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, RIT_testing_data)
                                                                                                                                  
                                   
                                                                          
                bar.next()  
                                                                                                                                                                                                                          
    else:
        # print('THIS ONE')
        bar = IncrementalBar('Randomly Gen Model Runs', max = Nsim)

        for i in range (0, Nsim):
            testing_startTime = datetime.now() 
            if parameter_settings.iloc[1, 1]!= '':
                frequency_of_screening_subpop_input=np.array(parameter_settings.iloc[1,1:Nsub+1]) #How often people are screened (in days)
                frequency_of_screening=np.zeros((Nsim, Nsub))
                for m in range(0, len(parameter_settings.iloc[1,1:Nsub+1])):
                    frequency_of_screening[:,m]=np.float(frequency_of_screening_subpop_input[m])
            else:
                frequency_of_screening= frequency_of_screening 
                
            
            (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
            isolation_pool_population, mat_subpop, mat_subpop_totals) = Randomly_Generated_Setup_And_Run_Model_Function(i, N, Nsim, Nsub, 
                                                                                                                        input_settings, parameter_settings, 
                                                                                                                        cycles_per_test_parameter, new_infections_per_shock_input, 
                                                                                                                        new_infections_per_shock_og, percent_subpop,frequency_of_screening,
                                                                                                                        frequency_ES, time_recovery, advancing_symptoms, Rt, 
                                                                                                                        symptom_case_fatality_ratio, days_to_incubation, 
                                                                                                                        time_to_return_fps, test_sensitivity, test_specificity, 
                                                                                                                        gamma_u, gamma_v, initial_infect, X, Y, max_iso_population_all_sims, 
                                                                                                                        max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims,
                                                                                                                        cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                                                                                                        vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, 
                                                                                                                        deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims,  
                                                                                                                        max_total_new_infect_2_days_all_sims,max_vaxxed_new_infect_2_days_all_sims,
                                                                                                                        max_unvaxxed_new_infect_2_days_all_sims, max_total_new_infect_week_all_sims,
                                                                                                                        max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, RIT_testing_data)
               
            bar.next()       
                                                                                                                                                                                 
if input_settings.iloc[21,1] == 1:    
    if input_settings.iloc[17,1] == 1 or input_settings.iloc[18,1] == 1 or input_settings.iloc[19,1] == 1:
        sim_num=np.zeros(Nsim)
        for i in range(0, Nsim):
            unvaxxed_deaths_all_sims[i] = results[i][0]
            vaxxed_deaths_all_sims[i] = results[i][1]
            deaths_all_sims[i] = results[i][2]
            max_iso_population_all_sims[i] = results[i][3]
            max_vaxxed_iso_population_all_sims[i] = results[i][4]
            max_unvaxxed_iso_population_all_sims[i] = results[i][5]
            cumulative_infections_all_sims[i] = results[i][6]
            cumulative_sympto_infections_all_sims[i] = results[i][7]
            vaxxed_sympto_infections_all_sims[i] = results[i][8]
            unvaxxed_sympto_infections_all_sims[i] = results[i][9]
            max_total_new_infect_2_days_all_sims[i] = results[i][10]
            max_vaxxed_new_infect_2_days_all_sims[i] = results[i][11]
            max_unvaxxed_new_infect_2_days_all_sims[i] = results[i][12]
            max_total_new_infect_week_all_sims[i] = results[i][13]
            max_vaxxed_new_infect_week_all_sims[i] = results[i][14]
            max_unvaxxed_new_infect_week_all_sims[i] = results[i][15]
            sim_num[i]=results[i][16]
else:
    print('Creating sim num vec outside function')
    sim_num=np.linspace(0, Nsim -1, Nsim)

            

   
#Saving Sobol Sensitivity Data    
if input_settings.iloc[19,1]==1: #signals we are collecting outputs of intereste for Sobol for other figures, this way we are not using time to run it when we do not need it
   Sobol_Data_Collection(input_settings, sim_num, param_values_test , max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, cumulative_sympto_infections_all_sims)

# if input_settings.iloc[18,1]==1:
#     Random_Forest_Data_Collection(Nsim, Nsub, frequency_of_screening, frequency_ES, time_recovery, advancing_symptoms, 
#                                       Rt, symptom_case_fatality_ratio, days_to_incubation, 
#                                       time_to_return_fps, test_specificity, test_sensitivity,
#                                       gamma_v_start, gamma_v_mod,  gamma_u, new_infections_per_shock_og,
#                                       max_total_new_infect_2_days_all_sims, max_vaxxed_new_infect_2_days_all_sims,
#                                       max_unvaxxed_new_infect_2_days_all_sims,max_total_new_infect_week_all_sims,
#                                       max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, 
#                                       max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, 
#                                       cumulative_sympto_infections_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
#                                       unvaxxed_sympto_infections_all_sims)

# Data_storage_paper_one_figures(max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, 
#                                     cumulative_sympto_infections_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
#                                     unvaxxed_sympto_infections_all_sims)
mat_subpop_totals_df = pd.DataFrame(mat_subpop_totals)
mat_subpop_totals_df.to_csv('mat_subpop_totals_cleanup_test_17.csv')
print('Total time for all simulations', datetime.now() - startTime) #prints how long the script took

# cumulative_infections_vec = round(cumulative_infect_finalR) * np.ones(N)
# benchmark_vals = np.column_stack((mat_subpop_totals, isolation_pool_population, cumulative_infections_vec))
# benchmark_vals_df = pd.DataFrame(benchmark_vals, columns=['Total Uninfected', 'Total False Positive', 'Total Exposed', 'Total Asympomatic', 'Total Symptomatic', 'Total True Positive', 'Total Recovered', 'Total Deceased', 'Isolation Population', 'Cumulative Infections'])
# benchmark_vals_df.to_csv('/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Benchmark Testing/New_Testing_Term_Calibration.csv')