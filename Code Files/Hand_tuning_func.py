#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:19:29 2023

@author: meghanchilds
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
from Param_and_Data_Organization_Funcs import VE_Benchmark_params, Benchmark_params, CSV_params,  Sobol_params, random_gen_params, calibrated_param_samples
from progress.bar import IncrementalBar
from Data_Collection_Scripts import Sobol_Data_Collection, RMSE_Calc_Data_Collection, Random_Forest_Data_Collection, Data_storage_paper_one_figures,  Metrics_of_Interest_Collection_Subpops, Metrics_of_Interest_Collection_Onepop, Testing_Data_Collection
from tqdm import tqdm, trange



def hand_tuning(frequency_ES_input, test_sensitivity_input, test_specificity_input, initial_infect_input, advancing_symptoms_input, Rt_input, new_infections_per_shock_input_ht):
    
    #Start timing script
    startTime = datetime.now()

    CSV_startTime = datetime.now()

    #Read in necessary CSVs

    parameter_settings=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Model_Parameter_Inputs.csv')
    parameter_settings=parameter_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
    input_settings=pd.read_csv (r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Model_Settings_Inputs.csv')   #reads the csv file with the model settings values
    input_settings=input_settings.replace(np.nan, '', regex=True) #Replaces empty cells with '' instead of nan so we can recognize it later in the code
    RIT_testing_datas=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Campus Data/Simplified_RIT_Data_Fall20.csv') #reads the csv file with RIT testing data
    RIT_testing_data=RIT_testing_datas.replace(np.nan, 0, regex=True) #Replaces empty cells with 0 instead of nan so we can use them to find the sum between the codes

    print('READ in CSVS',datetime.now() - CSV_startTime) 
    
    #Testing Inputs 
    testing_data_input=input_settings.iloc[13,1] #signals whether testing data is used for subpopulation model
    
    ##Global variables
    if input_settings.iloc[0,1]=='': #signifies we are running Sobol sensitivity analysis
        print('Sobol Global Variables')
        Num_params=12 # number of parameters we are running Sobol for 
        Nsim=int(input_settings.iloc[20,1]*(Num_params+2))
        N_Sobol=int(input_settings.iloc[20,1])
        
    elif input_settings.iloc[22,1] == 1: #signifies the data is being run with the calibrated samples
        
        calibrated_param_samples_csv=pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Calibrated Parameters/Fall_Samples_For_Analysis_FLF_300_sims_New_init_infect_SE_090_initial_infect_10.csv')
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
        input_settings.iloc[10,1] = RIT_testing_data.iloc[0,4]
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
    cycles_per_test_data=np.zeros((int(N), int(Nsub))) #Initializes empty matrix to fill
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
        
    LHS_startTime = datetime.now()
     
    #Generate Matrix of Randomly Generated Parameter sets using latin hypercube sampling## 
     
    print('Generating random parameters')
     
    #Latin Hypercube Sampling
     
    (frequency_of_screening, frequency_ES, time_recovery,  
     advancing_symptoms, Rt, symptom_case_fatality_ratio, 
     days_to_incubation, time_to_return_fps, test_specificity, 
     test_sensitivity,  gamma_v_start, gamma_v_mod,  gamma_u, 
     new_infections_per_shock_og, initial_infect) = random_gen_params(N, Nsub, Nsim, input_settings)
   
    print('Freq ES', frequency_ES, 'advance',  advancing_symptoms, 'Rt', Rt, 'SP', test_specificity, 'SE', test_sensitivity, 'New infect', new_infections_per_shock_og, 'initial infect', initial_infect)
    if new_infections_per_shock_input_ht == None:
        new_infections_per_shock_og = new_infections_per_shock_og
    else: 
        new_infections_per_shock_og = np.resize(np.array(new_infections_per_shock_input_ht), (1,1))
    print(new_infections_per_shock_input_ht)    
    print('LHS',datetime.now() - LHS_startTime)    
    
    if input_settings.iloc[16,1]==0: # checks if switch for decreasing efficacy is off
           
            print('Steady VE') # for deubbing

            gamma_v_end=np.zeros(len(gamma_v_start))

            for i in range (0,Nsim):
                gamma_v_end[i]=gamma_v_start[i] #sets gamma start to gamma end when we do not have decreasing efficacy
            
                Y[0,i]=gamma_v_start[i]
                Y[1,i]=gamma_v_end[i]
    X=np.array([0, N-1]) #sets our X values for our interpolation as our first and last indices 

                
    #Testing Data Case
    if testing_data_input == 1: # Checks to see if we are using testing data
        
        print('Running Testing Data Case')
        for i in trange(0,Nsim):
            print(i, Nsim)
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
            
            
            print('Randomly Generated Samples')
            if frequency_ES_input == None:
                frequency_ES = frequency_ES
                print(frequency_ES)
            else:
                frequency_ES = np.resize(np.array(frequency_ES_input), (1,1))
            
            if test_sensitivity_input == None:

                test_sensitivity = test_sensitivity
            else:
                test_sensitivity = np.resize(np.array(test_sensitivity_input), (1,1))
            
            if test_specificity_input == None:
                test_specificity = test_specificity
            else:
                test_specificity = np.resize(np.array(test_specificity_input), (1,1))
                print(type(test_specificity))
                
            if initial_infect_input == None:
                initial_infect = initial_infect
            else:
                initial_infect = np.resize(np.array(initial_infect_input), (1,1))
                
            if advancing_symptoms_input == None:
                advancing_symptoms = advancing_symptoms
            else:
                advancing_symptoms = np.resize(np.array(advancing_symptoms_input), (1,1))
            if Rt_input == None:
                Rt = Rt
            else:
                Rt = np.resize(np.array(Rt_input), (1,1))
            
            print(frequency_ES)
            parameters=np.vstack((frequency_ES[i,:], (time_recovery[i,:]), (advancing_symptoms[i,:]), 
                                      (Rt[i,:]), (symptom_case_fatality_ratio[i,:]), (days_to_incubation[i,:]), 
                                      (time_to_return_fps[i,:]), (test_sensitivity[i,:]), (test_specificity[i,:]),
                                      (gamma_u[i,:]), initial_infect[i,:])) 
            print(parameters[0,:])
                
            new_infections_per_ES_input=new_infections_per_shock_input
                
            settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
            
            vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
            
            # print('storing parameters',datetime.now() - parameters_startTime) 
            
            testing_data=cycles_per_test_data #if we not are we set the testing data to be the one calculated using a parameter (either CSV input or randomly generated)
            
            diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
            
            for m in range (0,N):
                gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            
            
            model_startTime = datetime.now()
            
            if Nsub ==1 and input_settings.iloc[18,1]:
                ( total_new_infections_per_2_days, total_new_infections_per_week, 
                       max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                       cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                       cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                       cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                       new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                       isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, testing_data, 
                                                                                         percent_subpop, new_infections_per_ES_input, 
                                                                                         vaccination_status, gamma_v, RIT_testing_data, Nsim)
            else:
                  (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                  isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings, testing_data, 
                                                                                       percent_subpop, new_infections_per_ES_input, 
                                                                                       vaccination_status, gamma_v, RIT_testing_data, Nsim)
            #Data Comparison
            if testing_data_input == 1: #only does calculation if we are using testing data
                    
                    fp_per_day = np.zeros(int(N/3))
                    tp_per_day = np.zeros(int(N/3))
                    sympt_per_day = np.zeros(int(N/3))
                    
                    for k in range(0, int(N/3)):
                        isolation_pool[k]=np.mean((isolation_pool_population[3*k], isolation_pool_population[(3*k)+1], isolation_pool_population[(3*k)+2])) #totals complete isolation pool for each day  
                        fp_per_day[k] = np.mean((mat_subpop_totals[3*k, 1], mat_subpop_totals[(3*k)+1, 1], mat_subpop_totals[(3*k)+2, 1]))
                        tp_per_day[k] = np.mean((mat_subpop_totals[3*k, 5], mat_subpop_totals[(3*k)+1, 5], mat_subpop_totals[(3*k)+2, 5]))
                        sympt_per_day[k] = np.mean((mat_subpop_totals[3*k, 4], mat_subpop_totals[(3*k)+1, 4], mat_subpop_totals[(3*k)+2, 4]))
                        
            plt.figure()
            days_F_iso = np.linspace(1, int(N/3), int(N/3))
            plt.bar(days_F_iso, RIT_testing_data.iloc[:,4])
            plt.plot(days_F_iso, isolation_pool, color = 'black')
            plt.xlabel('days')
            plt.ylabel('Iso Pop')
            plt.axvline(x = 43, color = 'red', linestyle = 'dashdot')
            plt.axvline(x = 54, color = 'red', linestyle = 'dashdot')
            plt.axvline(x = 66, color = 'red', linestyle = 'dashdot')
            plt.show()
            print('Total time for all simulations', datetime.now() - startTime) #prints how long the script took
            
            plt.figure()
            plt.bar(days_F_iso, fp_per_day, label ='FP ')
            plt.bar(days_F_iso, tp_per_day, bottom = fp_per_day, label = 'TP')
            plt.bar(days_F_iso, sympt_per_day, bottom = fp_per_day+tp_per_day, label ='S')
            plt.plot(days_F_iso, isolation_pool, label = ' iso pop total')
            plt.legend()
            plt.xlabel('days')
            plt.ylabel('People')
            plt.title('Iso Pop Broken Down')
            plt.show()
            return(mat_subpop_totals)
            
