#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:04:27 2022

@author: meghanchilds
"""

import numpy as np
import pandas as pd 
from pyDOE import *
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform, beta, truncnorm
import math 
from progress.bar import Bar, IncrementalBar
from tqdm import tqdm, trange

# Functions to generate param values

def VE_Benchmark_params(N, Nsub, Nsim, VE_Benchmark_parameter_settings):
    
    # Pulls param values from VE Benchmark CSV
    
    frequency_ES=VE_Benchmark_parameter_settings.iloc[0,1:Nsub+1]
    time_recovery=VE_Benchmark_parameter_settings.iloc[1,1:Nsub+1]
    advancing_symptoms=VE_Benchmark_parameter_settings.iloc[2,1:Nsub+1]
    Rt=VE_Benchmark_parameter_settings.iloc[3,1:Nsub+1]
    symptom_case_fatality_ratio=VE_Benchmark_parameter_settings.iloc[4,1:Nsub+1]
    days_to_incubation=VE_Benchmark_parameter_settings.iloc[5,1:Nsub+1]
    time_to_return_fps=VE_Benchmark_parameter_settings.iloc[6,1:Nsub+1]
    new_infections_per_shock_og=VE_Benchmark_parameter_settings.iloc[11,1:Nsub+1]
    print(new_infections_per_shock_og)
    test_specificity=VE_Benchmark_parameter_settings.iloc[8,1:Nsub+1]
    test_sensitivity=VE_Benchmark_parameter_settings.iloc[7,1:Nsub+1]
    gamma_v_start=VE_Benchmark_parameter_settings.iloc[9,1]
    gamma_u=VE_Benchmark_parameter_settings.iloc[10,1:Nsub+1]
    
    # new_infections_per_shock_og=np.zeros((Nsub,Nsim))
    # for i in range (0,Nsub):
    #     new_infections_per_shock_og[:,i]=new_infections_per_shock_input[i]
    
    return(frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_u, new_infections_per_shock_og)

def Benchmark_params(N, Nsub, Nsim, benchmark_parameter_settings):
    
    # Pulls param values from Benchmark CSV
    
    frequency_ES=benchmark_parameter_settings.iloc[0,1:Nsub+1]
    time_recovery=benchmark_parameter_settings.iloc[2,1:Nsub+1]
    advancing_symptoms=benchmark_parameter_settings.iloc[3,1:Nsub+1]
    Rt=benchmark_parameter_settings.iloc[4,1:Nsub+1]
    symptom_case_fatality_ratio=benchmark_parameter_settings.iloc[5,1:Nsub+1]
    days_to_incubation=benchmark_parameter_settings.iloc[6,1:Nsub+1]
    time_to_return_fps=benchmark_parameter_settings.iloc[7,1:Nsub+1]
    new_infections_per_shock_og=benchmark_parameter_settings.iloc[8,1:Nsub+1]
    test_specificity=benchmark_parameter_settings.iloc[9,1:Nsub+1]
    test_sensitivity=benchmark_parameter_settings.iloc[10,1:Nsub+1]
    gamma_v_start=benchmark_parameter_settings.iloc[11,1:Nsub+1]
    gamma_u=benchmark_parameter_settings.iloc[11,1:Nsub+1]
    
    return(frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_u, new_infections_per_shock_og)

def testing_term_benchmark_params(N, Nsub, Nsim, parameter_settings):
   
    # Pulls param values from Model_Parameter CSV
   
    frequency_ES = parameter_settings.iloc[0,1:Nsub+1]
    time_recovery = parameter_settings.iloc[2,1:Nsub+1]
    advancing_symptoms = parameter_settings.iloc[3,1:Nsub+1]
    Rt = parameter_settings.iloc[4,1:Nsub+1]
    symptom_case_fatality_ratio = parameter_settings.iloc[5,1:Nsub+1]
    days_to_incubation = parameter_settings.iloc[6,1:Nsub+1]
    time_to_return_fps = parameter_settings.iloc[7,1:Nsub+1]
    new_infections_per_shock_og = parameter_settings.iloc[8,1:Nsub+1]
    test_specificity = parameter_settings.iloc[9,1:Nsub+1]
    test_sensitivity = parameter_settings.iloc[10,1:Nsub+1]
    gamma_v_start = parameter_settings.iloc[11,1:Nsub+1]
    gamma_u  =parameter_settings.iloc[11,1:Nsub+1]  
    delay = parameter_settings.iloc[15,1:Nsub+1]  
    non_compliance = np.zeros(Nsub)

    
    return(frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_u, new_infections_per_shock_og, delay, non_compliance)


def CSV_params(N, Nsub, Nsim, parameter_settings):
   
    # Pulls param values from Model_Parameter CSV
   
    frequency_ES = parameter_settings.iloc[0,1:Nsub+1]
    time_recovery = parameter_settings.iloc[2,1:Nsub+1]
    advancing_symptoms = parameter_settings.iloc[3,1:Nsub+1]
    Rt = parameter_settings.iloc[4,1:Nsub+1]
    symptom_case_fatality_ratio = parameter_settings.iloc[5,1:Nsub+1]
    days_to_incubation = parameter_settings.iloc[6,1:Nsub+1]
    time_to_return_fps = parameter_settings.iloc[7,1:Nsub+1]
    new_infections_per_shock_og = parameter_settings.iloc[8,1:Nsub+1]
    test_specificity = parameter_settings.iloc[9,1:Nsub+1]
    test_sensitivity = parameter_settings.iloc[10,1:Nsub+1]
    gamma_v_start = parameter_settings.iloc[11,1:Nsub+1]
    gamma_u  =parameter_settings.iloc[11,1:Nsub+1]  
    delay = parameter_settings.iloc[15,1:Nsub+1]  
    non_compliance = parameter_settings.iloc[16,1:Nsub+1]  

    
    return(frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_u, new_infections_per_shock_og, delay, non_compliance)

def Sobol_params(N, Nsub, Nsim, N_Sobol, Num_params):
    
    frequency_ES_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    time_recovery_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    advancing_symptoms_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    Rt_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    symptom_case_fatality_ratio_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    days_to_incubation_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    time_to_return_fps=np.ones((2*N_Sobol,Nsub));
    new_infections_per_shock_og=lhs(1, samples=2*N_Sobol, criterion='center');
    test_specificity_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    test_sensitivity_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    frequency_of_screening_sample=lhs(1, samples=2*N_Sobol, criterion='center');
    gamma_v_start_og=lhs(1, samples=2*N_Sobol, criterion='center'); #0.1*np.ones((Nsim, Nsub))
    gamma_v_mod_sample=lhs(1, samples=2*N_Sobol, criterion='center'); #0.1*np.ones((Nsim, Nsub))
    gamma_u=np.ones((2*N_Sobol, Nsub));
    
    bar = IncrementalBar('Sobol Params Pt 1', max = 2*N_Sobol)
    
    for i in range(0,2*N_Sobol):
        new_infections_per_shock_og[i,:]=nbinom(5,0.25).ppf(new_infections_per_shock_og[i,:]);
        #gamma_v[:,i]=triang(0.8, loc=0, scale=1).ppf(gamma_v[:,i]); #high risk prior
        gamma_v_start_og[i,:]=triang(0.5, loc=0, scale=1).ppf(gamma_v_start_og[i,:]); #med risk prior
        #gamma_v[:,i]=triang(0.1, loc=0, scale=1).ppf(gamma_v[:,i]); #low risk prior
        gamma_v_mod_sample[i,:]=triang(0.3, loc=0, scale=1).ppf(gamma_v_mod_sample[i,:]);
    
        bar.next()
        
    frequency_ES=np.zeros((2*N_Sobol, Nsub))
    time_recovery=np.zeros((2*N_Sobol, Nsub))
    advancing_symptoms=np.zeros((2*N_Sobol, Nsub))
    Rt=np.zeros((2*N_Sobol, Nsub))
    symptom_case_fatality_ratio=np.zeros((2*N_Sobol, Nsub))
    days_to_incubation=np.zeros((2*N_Sobol, Nsub))
    test_specificity=np.zeros((2*N_Sobol, Nsub))
    test_sensitivity=np.zeros((2*N_Sobol, Nsub))
    frequency_of_screening=np.zeros((2*N_Sobol, Nsub))
        
    bar = IncrementalBar('Sobol Params Pt 2', max = 2*N_Sobol)

    for i in range (0,2*N_Sobol):
         frequency_ES[i,:]=binom(14,0.5).ppf(frequency_ES_sample[i,:]);
         time_recovery[i,:]=triang(4/11, loc=10, scale=11).ppf(time_recovery_sample[i,:]);
         advancing_symptoms[i,:]=triang(5/9, loc=5, scale=45).ppf(advancing_symptoms_sample[i,:]);
         Rt[i,:]=triang(0.12, loc=0.8, scale=1.7).ppf(Rt_sample[i,:]);
         symptom_case_fatality_ratio[i,:]=triang(0.05, loc=0, scale=0.01).ppf(symptom_case_fatality_ratio_sample[i,:]);
         days_to_incubation[i,:]=triang(2/9, loc=3, scale=9).ppf(days_to_incubation_sample[i,:]);
         test_sensitivity[i,:]=triang(0.5, loc=0.7, scale=0.2).ppf(test_sensitivity_sample[i,:]);
         test_specificity[i,:]=triang(0.6,loc=0.95, scale=0.05).ppf(test_specificity_sample[i,:]);
         frequency_of_screening[i,:]=nbinom(5, 0.25).ppf(frequency_of_screening_sample[i,:]);
         
         bar.next()
         
    #print(frequency_of_screening)   
    print(len(frequency_ES))
    params=np.column_stack((frequency_ES[:,0], time_recovery[:,0], advancing_symptoms[:,0], Rt[:,0], symptom_case_fatality_ratio[:,0], 
                            days_to_incubation[:,0],  test_sensitivity[:,0], test_specificity[:,0], new_infections_per_shock_og,  
                            frequency_of_screening[:,0],  gamma_v_start_og, gamma_v_mod_sample))
    param_values_test=np.zeros((N_Sobol*(Num_params+2),(Num_params)))

    bar = IncrementalBar('Sobol Params Pt 3', max = Num_params)
    

    for i in range(0,Num_params): # goes across all columns, one for each parameter
         for j in range (0, N_Sobol): # goes through all sets of rows
            param_values_test[(Num_params+2)*j:(i+1)+(Num_params+2)*j,i]=params[j,i]
            # param_values_test[(12*j)+(i+2),i]=params[j,i]
            param_values_test[((Num_params+2)*j)+(i+1),i]=params[j+int(N_Sobol/2),i]
            param_values_test[((Num_params+2)*j)+(Num_params+1),i]=params[j+int(N_Sobol/2),i]
            param_values_test[((Num_params+2)*j)+(i+2):((Num_params+2)*j)+(Num_params+1),i]=params[j,i]
              
            bar.next()
            
    gamma_v_start=param_values_test[:,10] #pulls gamma_v_start from matrix for later calculations
    new_infections_per_shock_og=param_values_test[:,8] #pulls new_infections_per_shock from matrix for later calculations
    gamma_v_mod=param_values_test[:,11]  #pulls gamma_v_end from matrix for later calculations  
    
    return(param_values_test, gamma_v_start, new_infections_per_shock_og, gamma_v_mod, time_to_return_fps, gamma_u)

def random_gen_params(N, Nsub, Nsim, input_settings):
    
# =============================================================================
#     ### LHS SAMPLING##
# =============================================================================
    
    frequency_ES_sample = lhs(1, samples=Nsim, criterion='center');
    time_recovery_sample = lhs(1, samples=Nsim, criterion='center');
    advancing_symptoms_sample = lhs(1, samples=Nsim, criterion='center');
    Rt_sample=lhs(1, samples = Nsim, criterion='center');
    symptom_case_fatality_ratio_sample = lhs(1, samples=Nsim, criterion='center');
    days_to_incubation_sample = lhs(1, samples=Nsim, criterion='center');
    time_to_return_fps = np.ones((Nsim,Nsub));
    new_infections_per_shock_og = lhs(1, samples=Nsim, criterion='center');
    test_specificity_sample = lhs(1, samples=Nsim, criterion='center');
    test_sensitivity_sample = lhs(1, samples=Nsim, criterion='center');
    gamma_v_start = lhs(1, samples=Nsim, criterion='center'); #0.1*np.ones((Nsim, Nsub))
    gamma_v_mod = lhs(1, samples=Nsim, criterion='center'); #0.1*np.ones((Nsim, Nsub))
    gamma_u=np.ones((Nsim, Nsub));
    frequency_of_screening_sample = lhs(1, samples=Nsim, criterion='center');
    initial_infect_sample = lhs(1, samples=Nsim, criterion='center');
    delay_sample = lhs(1, samples=Nsim, criterion='center');
    non_compliance_sample = lhs(1, samples=Nsim, criterion='center');


    #bar = IncrementalBar('Random Params Pt 1', max = Nsim)
    
    for i in trange(0,Nsim):
        new_infections_per_shock_og[i,:]=nbinom(5,0.25).ppf(new_infections_per_shock_og[i,:]);
        #gamma_v_start[i,:]=triang(0.8, loc=0, scale=1).ppf(gamma_v_start[i,:]); #high risk prior
        gamma_v_start[i,:]=triang(0.5, loc=0, scale=1).ppf(gamma_v_start[i,:]); #med risk prior
        #gamma_v_start[i,:]=triang(0.1, loc=0, scale=1).ppf(gamma_v_start[i,:]); #low risk prior
        gamma_v_mod[i,:]=triang(0.3, loc=0, scale=1).ppf(gamma_v_mod[i,:])


        #bar.next()
    if input_settings.iloc[23, 1] == 1: # triggers if we are not using the vaccine efficacy parmater
        gamma_v_start = np.ones(Nsim)   
    
    frequency_ES=np.zeros((Nsim, Nsub))
    time_recovery=np.zeros((Nsim, Nsub))
    advancing_symptoms=np.zeros((Nsim, Nsub))
    Rt=np.zeros((Nsim, Nsub))
    symptom_case_fatality_ratio=np.zeros((Nsim, Nsub))
    days_to_incubation=np.zeros((Nsim, Nsub))
    test_specificity=np.zeros((Nsim, Nsub))
    test_sensitivity=np.zeros((Nsim, Nsub))
    #frequency_of_screening=np.zeros((Nsim, Nsub)) 
    frequency_of_screening = np.zeros((Nsim, Nsub))
    initial_infect = np.zeros((Nsim, Nsub))
    delay = np.zeros((Nsim,1))
    non_compliance = np.zeros((Nsim,Nsub))

    
    #bar = IncrementalBar('Random Params Pt 2', max = Nsim)
    for i in trange (0, Nsim):
        frequency_ES[i,:]=binom(14,0.5).ppf(frequency_ES_sample[i,:]);
        time_recovery[i,:]=triang(4/11, loc=10, scale=11).ppf(time_recovery_sample[i,:]);
        advancing_symptoms[i,:]=triang(5/9, loc=5, scale=45).ppf(advancing_symptoms_sample[i,:]);
        Rt[i,:]=triang(0.12, loc=0.8, scale=1.7).ppf(Rt_sample[i,:]);
        symptom_case_fatality_ratio[i,:]=triang(0.05, loc=0, scale=0.01).ppf(symptom_case_fatality_ratio_sample[i,:]);
        days_to_incubation[i,:]=triang(2/9, loc=3, scale=9).ppf(days_to_incubation_sample[i,:]);
        # test_specificity[i,:]=triang(c = 0.6, loc=0.95, scale=0.05).ppf(test_specificity_sample[i,:]);
        test_specificity[i,:]=norm(loc=1.23, scale=0.007).ppf(test_specificity_sample[i,:]);
        test_sensitivity[i,:]=triang(0.5, loc=0.7, scale=0.2).ppf(test_sensitivity_sample[i,:]);
        frequency_of_screening[i,:]=nbinom(5, 0.25).ppf(frequency_of_screening_sample[i,:]);
        initial_infect[i,:]=nbinom(12, 0.5).ppf(initial_infect_sample[i,:]);
        #delay[i,:]=binom(100,0.2).ppf(delay_sample[i,:]);
        # non_compliance[i,:] = truncnorm(-3.3, 3.3, loc = 0.05, scale = 0.015 ).ppf(non_compliance_sample[i,:])
    delay = np.zeros((Nsim,1))
    non_compliance = np.zeros((Nsim,1))
    

   
        
    return(frequency_of_screening, frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_v_mod,  gamma_u, new_infections_per_shock_og, initial_infect, delay, non_compliance)

def calibrated_param_samples(N, Nsub,input_settings, calibrated_param_samples_csv):
    # Pulls params from Calibration samples CSV or fixes parameters at specified values
    num_samples = len(calibrated_param_samples_csv.iloc[:,1]) # finds how many samples we saved so we can match length of fixed params
    frequency_ES  = np.array(calibrated_param_samples_csv.iloc[:,1] )
    frequency_ES.resize((num_samples, 1))
    time_recovery = np.array( calibrated_param_samples_csv.iloc[:,2]) 
    time_recovery.resize((num_samples, 1))
    advancing_symptoms = np.array(calibrated_param_samples_csv.iloc[:,3] )
    advancing_symptoms.resize((num_samples, 1))
    Rt =  np.array(calibrated_param_samples_csv.iloc[:,4])
    Rt.resize((num_samples, 1))
    symptom_case_fatality_ratio = np.array(calibrated_param_samples_csv.iloc[:,5])
    symptom_case_fatality_ratio.resize((num_samples, 1))
    days_to_incubation = np.array(np.transpose(calibrated_param_samples_csv.iloc[:,6]))
    days_to_incubation.resize((num_samples, 1))
    time_to_return_fps = np.ones((num_samples, Nsub)); # fixed at 1
    new_infections_per_shock_og = np.array(calibrated_param_samples_csv.iloc[:,9]) # fixed at 2 infections per exogenous shock
    new_infections_per_shock_og.resize((num_samples, 1))
    test_sensitivity = np.array(calibrated_param_samples_csv.iloc[:,7])
    test_sensitivity.resize((num_samples, 1))
    test_specificity = np.array(calibrated_param_samples_csv.iloc[:,8]) # p.ones(len(test_sensitivity))
    test_specificity.resize((num_samples, 1))
    delay =  np.zeros(num_samples)#np.array(calibrated_param_samples_csv.iloc[:,11]) # p.ones(len(test_sensitivity))
    delay.resize((num_samples, 1))
    non_compliance = np.zeros(num_samples) #np.array(calibrated_param_samples_csv.iloc[:,12]) # np.ones(len(test_sensitivity))
    non_compliance.resize((num_samples, 1))
    
    
    if input_settings.iloc[31, 1] == 1: # checks whether initial infected was randomly sampled, if yes then pulls those calibrated parameters
        initial_infect = np.array(calibrated_param_samples_csv.iloc[:,10])
        initial_infect.resize((num_samples, 1))
    else: # if not then we use zeroes as this term will be ignored
        initial_infect = np.zeros_like(test_sensitivity)
    
    if input_settings.iloc[39,1] == 1: # if we are adding uncertainty to intial iso
        initial_iso = np.array(calibrated_param_samples_csv.iloc[:,11]) # p.ones(len(test_sensitivity))
        initial_iso.resize((num_samples, 1))
    else: # if not use zeros as it will get overwritten in run model
        initial_iso = np.zeros_like(test_sensitivity)
    
    gamma_v_start = np.ones(num_samples); # no vaccinations so fixed at 
    gamma_v_mod = np.ones(num_samples); # no vaccinations so fixed at 1
    gamma_u = np.ones((num_samples, Nsub)); #
    
    return(frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_specificity, test_sensitivity,  gamma_v_start, gamma_v_mod,  gamma_u, new_infections_per_shock_og, initial_infect, delay, non_compliance, initial_iso)

    
# Function to generate testing data in a form the model can read
## INCORRECT CALC NO LONGER USED
# def testing_data_conversion(N, Nsub, RIT_testing_data, input_settings):
    
#     cycles_per_test_data=np.zeros((int(N), int(Nsub))) #Initializes empty matrix to fill
#     tests_per_day=np.zeros(int((N-1)/3)) #Initializes empty vector to fill
#     tests_per_cycle=np.zeros(int(N)) #Initializes empty vector to fill
#     for k in range(0,int(((N-1)/3))):
#         tests_per_day[k]=RIT_testing_data.iloc[k,2] #Imports data for tests per day from the csv
#     for j in range(0,int(((N-1)/3))):
#         tests_per_cycle[3*j:3*j+3]=tests_per_day[j]/3 #converts daily a data to tests per time step
#     for h in range(0,N):
#         if tests_per_cycle[h]==0: # if no tests have occured on this day automatically set cycles_per_test to zero to avoid divide by zero error
#             cycles_per_test_data[h,:]=0
#         else:
#             cycles_per_test_data[h,:]=(input_settings.iloc[3,1])/(tests_per_cycle[h]) #Uses testing data to calculate cycles per test
#     return(cycles_per_test_data, tests_per_cycle, tests_per_day)

# Function to calculate testing rate

def Testing_Rate_calc(i, Nsim, N, Nsub, RIT_testing_data, mat_subpop_totals, tests_per_timestep):
       
    # tau = num tests per orevious time step / (U(t-1) + A(t-1) + E(t-1))
   
    
    testing_rate = tests_per_timestep[i-1]/(mat_subpop_totals[i-1,0]+mat_subpop_totals[i-1,3]+mat_subpop_totals[i-1,2])
        # now we have our testing rate but we need to put it in terms of cycles_per_test which is what the model takes
     # we know tau = 1/ cycles_per test so this calculation is as follows
     # We add in a divide by zero edit so we do not get that error
     
    if testing_rate == 0:
        cycles_per_tests_data = 0
    else:    
        cycles_per_tests_data = 1 / testing_rate # takes one of the testing rate because tau (the testing rate ) is equal to 1/cycles_per_test
    
    
    return (cycles_per_tests_data)

# Function to calculate RMSE outside of running model



def days_to_timesteps(day_num, total_days):
    # 3 is the cycles per day
   timesteps = np.zeros(total_days * 3)
   
   for i in range(total_days * 3):
        timesteps[i] = i
   print(timesteps)     
   return([timesteps[3 * day_num : 3 * (day_num + 1)]])

# This function is used to redorganize the RIT data so we do not have the weekend occurences of positive tests with no tests administered
# Instead we move positive test notifications form the weekends to the following Mondays

def RIT_Data_ReOrg(RIT_testing_data):
    # Now we initliaze some values and empty vectors we will need to fill or use in our loops
    tests_per_day = RIT_testing_data.iloc[:,2]
    num_positive_tests_in_data = RIT_testing_data.iloc[:,3] # number of positive tests per day from the RIT data
    
    updated_positive_tests = np.zeros(len(tests_per_day))
    
    
    num_occurences_pos_test_no_test = 0
    
    
    
    
    # Now we start by looping through the data and saving the data points where there is not an issue and flagging how many times we have the issue
    # where there are 0 tests done and a nonzero amount of positive tests returned
    
    for i in range(len(RIT_testing_data)):
        
        if tests_per_day[i] == 0 and num_positive_tests_in_data[i] != 0: # if the ocurrence is happening 
            
            updated_positive_tests[i] = 0 # we set the updated positive tests to zero
            
            num_occurences_pos_test_no_test += 1 # and keep tally of how many time this has happeneded
                
            
            
        else:
            updated_positive_tests[i] = num_positive_tests_in_data[i] # if it is not happening the number of positive tests remain the same
    
    num_updated_occurences_pos_test_no_test = 0 
    index = 0                
    index_track = np.zeros(num_occurences_pos_test_no_test)
    redistributed_pos_results = np.zeros((num_occurences_pos_test_no_test,2)) # now we create a new empty matrix that has the same number of rows as the number of times  
    # that the number of tests per day was zero and the number of positve tests was non-zero, and has a column for the sum of the positive tests on the days this happens
    # Aand for the index this is happening at
    
    # Now we cycle again checking for this occurence and when we find it we keep track of index where it occurs and we add the test cases where this is occurring
    
    for i in range(len(RIT_testing_data)):
          
        if tests_per_day[i] == 0 and num_positive_tests_in_data[i] != 0: # first we again check where this is happening
              
              # print('first', i)
              
              for j in range(i, len(RIT_testing_data)): # then we start at this timestep and cycle forward 
                 
                  if tests_per_day[j] == 0 and num_positive_tests_in_data[j] != 0 and index_track[index - 1] != j: # This happedns while this is true AND it is 
                 # not happening at the last index this occurred at. This prevents double counting
                      # print('second', j)
                      index_track[index] = j # here we keep track of the last index it is occuring at so we do not double count this index
                      
                      redistributed_pos_results[index, 1] += num_positive_tests_in_data[j] # here we sum the num of positive tests while no tests are being done
                      redistributed_pos_results[index, 0] = i # here we note the first index this occurs at
    
                  else:
              
                     break # when there are not no tests done AND a nonzero number of positive tests we break this loop and go back to our main loop
                
              index += 1 # here we add one to our index so we move to the next entry in redistributed_po_results the next time this occurs
              
    
    redistributed_pos_results = redistributed_pos_results[~np.all(redistributed_pos_results== 0, axis=1)]

    # Now we input the new sum of these positive test results to the following monday
    for i in range(len(redistributed_pos_results[:,0])):
        
        index_of_interest = int(redistributed_pos_results[i,0]) # we keep track of the index where it started
        # we use this index to place the new positive tests reults with the positive tests results at two days later (monday)
       
        # we add the positive tests reults at this day with the results from the weekend
        updated_positive_tests[index_of_interest - 1] = num_positive_tests_in_data[index_of_interest - 1] + redistributed_pos_results[i,1]
     
        

    # then we check if we have any of this occurences to make sure we got them all
    for i in range(len(RIT_testing_data)):
        
        if tests_per_day[i] == 0 and updated_positive_tests[i] != 0:
            
            print('Fail')
            
            num_updated_occurences_pos_test_no_test += 1

    return(updated_positive_tests)

def Total_Positive_Tests_per_Day(i, j, probability_positive_test_U_A, U_A_pop_proportion, test_sensitivity, test_specificity, mat_subpop_totals, RIT_testing_data, settings):
   
    # if it is first timestep then everything is zero because we base out calculations on the timstep previous and cannot do that for stochastic
    if i == 0:
        true_pos_test = 0 
        false_pos_test = 0
        num_pos_tests_model = 0
        probability_positive_test_per_day_U_A = 0
        num_tests_U_A = 0
        U_A_pop_proportion_final = 0
        # print('stochastic', i/3, 'U_A', U_A_pop_proportion_final, 'tests', 0, 'tests adapted', num_tests_U_A, )
        # print('stochastic', 0, probability_positive_test_per_day_U_A)
        proba_pos_beta_check = np.resize(np.array(probability_positive_test_per_day_U_A), (1,1))
        df_proba_pos_beta_check  = pd.DataFrame(proba_pos_beta_check )

        df_proba_pos_beta_check.to_csv(r'/Users/meghanchilds/Desktop/proba_beta_check.csv', mode = 'a')

        if settings[32] ==1:
            n_check = np.resize(np.array(num_tests_U_A), (1,1))
            proba_pos_beta_check = np.resize(np.array(probability_positive_test_per_day_U_A), (1,1))
            
            df_n_check = pd.DataFrame(n_check)
            df_proba_pos_beta_check  = pd.DataFrame(proba_pos_beta_check )
            
            df_n_check.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Hand Tuning Outputs/n_check_recitaion_demo.csv',  header=False, mode = 'a')
            df_proba_pos_beta_check.to_csv(r'/Users/meghanchilds/Desktop/proba_pos_beta_check_recitaion_demo.csv',  header=False, mode = 'a')
    
    # now we set up calculations for all other timesteps
    else:
       
        # We set up a catch in case we try to run this with the subpop model in the future so we can
        # check in case the test sensitivity or specificity values are different
        if j > 0:
            print('Running stochastic test results with subpop model, check for accuracy, AKA are Se and Sp same for all subpops')
        
        # First we define variables we need
        test_sensitivity = test_sensitivity[0] # these will be the same for both subpopulations so set them as a scalar since we are not summing over subpops but are using total values
        test_specificity = test_specificity[0]
        
        tests_per_day = RIT_testing_data.iloc[:,2] # pulls num,ber of tests RIT does each day
        
       
        # Then we calculate values we need in terms of timesteps
        for m in range(int(i)-3, i):
            # print(m)
            # first we calculate the proportion of the transmission and testing class population that U and A make up
            # this will be used later to calculate the number of tests admisietred. We use this because we only account
            # for tests administered to the Uninfected and Asymptomatic compartments, because we assume all individuals
            # who have been exposed, test negative. This is the proportion at every timestep
            U_A_pop_proportion[m] = (mat_subpop_totals[m, 3] + mat_subpop_totals[m, 0])/( mat_subpop_totals[m, 0] +  mat_subpop_totals[m,2] +  mat_subpop_totals[m,3])
            # print('stochastic A', m, mat_subpop_totals[m, 3])
            # Now we calculate the probability of a positive tests at each timestep.
            # this first accounts for the zero timestep which has to be zero because we calculate the probability, based on the timestep before,
            # because we assume it takes 8 hours or one timestpe for positive results to come back.
            # if m == 0:
            #     print('m=0 triggered')
            #     probability_positive_test_U_A[0] = 0
            # # this calculates the probability at the rest of the timestep
            # else:
            probability_positive_test_U_A[m] = test_sensitivity *  (mat_subpop_totals[m,3] / ( mat_subpop_totals[m,0]  +  mat_subpop_totals[m,3] + mat_subpop_totals[m,2])) + (1 - test_specificity) * (mat_subpop_totals[m,0] / ( mat_subpop_totals[m,0] +  mat_subpop_totals[m,3]+ mat_subpop_totals[m,2])) 
            # print('stochastic', 'Se:', test_sensitivity, 'A:', mat_subpop_totals[m-1,3], 'U:', mat_subpop_totals[m-1,0], 'E:', mat_subpop_totals[m-1,2], '1-Sp:',  1 - test_specificity)
            
        
        # Then combine to get values in terms of days
        # We originally calculate eveyrhing in terms of timesteps as that is the timescale we use for the model on, but in order to apply the data we need eveything
        # in days as that is the timescale the data is uses
        probability_positive_test_U_A_df = pd.DataFrame(probability_positive_test_U_A)
        probability_positive_test_U_A_df.to_csv(r'/Users/meghanchilds/Desktop/proba_beta_check_per_timestep.csv')
        # For probability per day we find the mean of the probabilities over the three timesteps of the day
        probability_positive_test_per_day_U_A =  np.mean((probability_positive_test_U_A[int(i - 3)],  probability_positive_test_U_A[int(i - 2)],  probability_positive_test_U_A[int(i - 1)]))
        # print('stochastic', i/3, probability_positive_test_per_day_U_A)
        # resizes so we can save to dataframe
        probability_positive_test_per_day_U_A_resize = np.resize(np.array(probability_positive_test_per_day_U_A), (1,1))
        # saves value to dataframe and then to a csv for debugging and accuracy verification
        probability_positive_test_per_day_U_A_df = pd.DataFrame(probability_positive_test_per_day_U_A_resize)
        probability_positive_test_per_day_U_A_df.to_csv(r'/Users/meghanchilds/Desktop/proba_beta_check.csv', mode = 'a')
       
        # To find the proportion of the population that U and A occupy for the whole day we take the average of their proportions at each timestep of the day
        U_A_pop_proportion_final = np.mean(( U_A_pop_proportion[int(i - 3)], U_A_pop_proportion[int(i - 2)], U_A_pop_proportion[int(i - 1)]))
        
        # we also want the populations per day for the uninfected, asymptomatic, and exposed compartments. To find this we take the means of these
        # populations at the three timesteps that make up the day
        U_pop_per_day = np.mean((mat_subpop_totals[int(i - 3), 0], mat_subpop_totals[int(i - 2), 0],  mat_subpop_totals[int(i - 1), 0]))
        A_pop_per_day = np.mean(( mat_subpop_totals[int(i - 3), 3], mat_subpop_totals[int(i - 2), 3], mat_subpop_totals[int(i - 1), 3]))
        E_pop_per_day = np.mean((mat_subpop_totals[int(i - 3), 2],  mat_subpop_totals[int(i - 2), 2] , mat_subpop_totals[int(i - 1), 2]))
      
        # Then we calculate the number of tests administered for the binomial/beta distribution
        if test_sensitivity ==0 and test_specificity ==1: # case where no positive tests will come back
            # print('both trigger')
            num_tests_U_A = 0
        elif test_sensitivity == 0: # case where only tests administered to U will come back
            # print('se trigger')
            num_tests_U_A =  (U_pop_per_day/(U_pop_per_day+A_pop_per_day+E_pop_per_day))* tests_per_day[int(i/3)]
        elif test_specificity == 1: # case where only tests administered to A will come back
            # print('sp trigger')
            num_tests_U_A =  (A_pop_per_day/(U_pop_per_day+A_pop_per_day+E_pop_per_day))* tests_per_day[int(i/3)]
       
        else: # normal case where positive tests will come back from both
            # print('normal trigger')
            num_tests_U_A = (U_A_pop_proportion_final * tests_per_day[int(i/3)] )
        # print('stochastic', i/3, 'U_A', U_A_pop_proportion_final, 'tests', tests_per_day[int(i/3)- 1], 'tests adapted', num_tests_U_A, )
        # sets up checks to make sure tests administered do not go outside of normal bounds
# =============================================================================
#         # in this case the bounds are the total tests administered to U, A, and E and 0 as we cannot have a negative number of tests
# =============================================================================
        if num_tests_U_A > tests_per_day[int(i/3)]: # checks to make sure we do not have mosre tests between U and A then the total number of tests we actually administered
            print('ERROR: U AND A TESTS SUM TO GREATER THAN TOTAL TESTS', U_A_pop_proportion_final)
        if num_tests_U_A <  0:
            print('ERROR: Number of tests to U and A are Negative', num_tests_U_A, 'prop', U_A_pop_proportion_final, 'tests', tests_per_day[int(i/3)- 1], 'U',  mat_subpop_totals[i-3, 0],  mat_subpop_totals[i-2, 0],  mat_subpop_totals[i-1, 0], 'E', mat_subpop_totals[i-3, 2], mat_subpop_totals[i-2, 2], mat_subpop_totals[i-1, 2], 'A', mat_subpop_totals[i-3,3], mat_subpop_totals[i-2,3], mat_subpop_totals[i-2,3])
        
        # saves values to csv for debugging and accuracy validation
        if settings[32] ==1:
            n_check = np.resize(np.array(num_tests_U_A), (1,1))
            test_check = np.resize(np.array(tests_per_day[int(i/3)]), (1,1))
            proba_pos_beta_check = np.resize(np.array(probability_positive_test_per_day_U_A), (1,1))
            
            df_n_check = pd.DataFrame(n_check)
            df_test_check = pd.DataFrame(test_check)
            df_proba_pos_beta_check  = pd.DataFrame(proba_pos_beta_check )
          
            df_n_check.to_csv(r'/Users/meghanchilds/Desktop/n_check_recitaion_demo.csv',  header=False, mode = 'a')
            df_test_check.to_csv(r'/Users/meghanchilds/Desktop/test_check_recitaion_demo.csv',  header=False, mode = 'a')
            df_proba_pos_beta_check.to_csv(r'/Users/meghanchilds/Desktop/proba_pos_beta_check_recitaion_demo.csv',  header=False, mode = 'a')

        # Then we calculate the theoretical params for the binomial and the alpha and beta variables for the beta dist
        
        # first the mean of the binmial we are approximating
        mean_bin =  num_tests_U_A * probability_positive_test_per_day_U_A
        
        # next using the mean of the binomial we calculate the alpha dn beta variables we use for the beta distirution that approximates our binomial dist
        alpha_var = mean_bin + 1
        beta_var = num_tests_U_A  - mean_bin + 1

        # now we set a statetment that catches if no tests should be coming back, i.e. Se = 0% and Sp = 100% then hardcodes num of positive tests to 0
        if test_sensitivity == 0 and test_specificity ==1: # case where teste sensitivity is 0% and test specificity is 100%, making sure we are robust to error
            num_pos_tests_model = 0
        # For all other cases it uses a beta approximation of a binomial to draw the number of positive tests in the model
        else: # all other cases
            #num_pos_tests_model = beta.rvs(a = alpha_var, b = beta_var, loc = 0, scale =  num_tests_U_A )
            num_pos_tests_model = binom.rvs(n = tests_per_day[int(i/3)], p =  probability_positive_test_per_day_U_A, size = 1)
        # print('stochastic', int(i/3), tests_per_day[int(i/3)], probability_positive_test_per_day_U_A)
        # splits up number of positive tests between true and false positive
        (true_pos_test, false_pos_test) = True_and_False_Positive(i, num_pos_tests_model, probability_positive_test_per_day_U_A, 
                                                                  test_sensitivity, test_specificity, E_pop_per_day, U_pop_per_day, A_pop_per_day)
        # print('HERE', num_pos_tests_model, true_pos_test, false_pos_test)
        # checks to make sure the number of true positive tests does not exceed the number of tests administered to the asymptomatic population
        if true_pos_test > (A_pop_per_day/(A_pop_per_day + U_pop_per_day + E_pop_per_day)) * tests_per_day[int(i/3)]:  # checks to make sure we do not have more true positive test results than we do number of tests adminsitered that could be true positive
            print('ERROR: More true positive test results than tests administered')
            print(i,(A_pop_per_day/(A_pop_per_day + U_pop_per_day + E_pop_per_day)) * tests_per_day[int(i/3)], true_pos_test, tests_per_day[int(i/3)] )
        
        # checks to make sure the number of false positive tests does not exceed the number of tests administered to the Uninfected populaion
        if false_pos_test > (U_pop_per_day/(A_pop_per_day + U_pop_per_day + E_pop_per_day)) * tests_per_day[int(i/3)]: # checks to make sure we do not have more false positive test results than we do number of tests adminsitered that could be false positive
            print('ERROR: More false positive test results than tests administered')
            print(i,(U_pop_per_day/(A_pop_per_day + U_pop_per_day + E_pop_per_day)) * tests_per_day[int(i/3)], false_pos_test, tests_per_day[int(i/3)] )
        
        # checks to make sure the total number of tests does not exceed the number of false and true positive
        if num_pos_tests_model - (false_pos_test + true_pos_test) > 10**-14:
            print('ERROR: FP and TP sum to greater that total',false_pos_test, true_pos_test, num_pos_tests_model)
        #print(num_tests_U_A, true_pos_test, false_pos_test, true_pos_test+false_pos_test)
    
    # Save to CSV for debugging
    tests_and_probability = np.column_stack((true_pos_test, false_pos_test, num_pos_tests_model, probability_positive_test_per_day_U_A, num_tests_U_A, U_A_pop_proportion_final))
    tests_and_probability_df = pd.DataFrame(tests_and_probability , columns = ['True positive', 'False positive', 'Total positive', 'Proba of positive', 'Number of Tests', 'U_A Prop'] )
    tests_and_probability_df.to_csv(r'/Users/meghanchilds/Desktop/Tests_and_Probability_Debug_stochasticity_redo_rerun.csv', mode = 'a')
    
    return(num_pos_tests_model, true_pos_test, false_pos_test, probability_positive_test_U_A, U_A_pop_proportion)

def True_and_False_Positive(i, num_pos_tests_model, probability_positive_test_per_day_U_A, test_sensitivity, test_specificity, E_pop_per_day, U_pop_per_day, A_pop_per_day):
    
    # checks for cases where only one type of test would come back
    
    # checks for case where only true positive tests come back
    if test_specificity == 1: # in case of 100% test specificity there would be no FP only TP
        true_pos_test = num_pos_tests_model
        false_pos_test = 0
    
    # checks for case where only false positive tests come back
    elif test_sensitivity == 0:  # in case of 100% test sensitivity there would be no TP only FP
        true_pos_test = 0
        false_pos_test = num_pos_tests_model
    
    # in all    
    else: # for all other cases of test sensitivity and specificity we split them FP and TP up proportionally  to their repsective populations
        true_pos_test = num_pos_tests_model * (( A_pop_per_day)/(A_pop_per_day  + U_pop_per_day))
        #true_pos_test = num_pos_tests_model * (((test_sensitivity * A_pop_per_day)/(A_pop_per_day  + U_pop_per_day))/ probability_positive_test_per_day_U_A)
    
        false_pos_test = num_pos_tests_model * ((U_pop_per_day )/(A_pop_per_day  + U_pop_per_day))
        #false_pos_test = num_pos_tests_model * ((((1 - test_specificity) * ((U_pop_per_day ))/(A_pop_per_day  + U_pop_per_day)))/ probability_positive_test_per_day_U_A)
    
    #print('Num Tests:',num_pos_tests_model, 'TP:', true_pos_test, 'FP:', false_pos_test  )
    
    return(true_pos_test, false_pos_test)

