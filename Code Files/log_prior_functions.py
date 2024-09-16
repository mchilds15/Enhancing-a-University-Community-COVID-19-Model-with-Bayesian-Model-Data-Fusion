#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:59:24 2022

@author: meghanchilds
"""
import numpy as np
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform, truncnorm
from RMSE_func import RMSE_Func_calc
from Run_Model_Functions import run_model
import sys
import pandas as pd

# This file has all the functions used to calculate the negative log posterior score for the model calibration

def log_prior_pdf(params, param_dists):
    
    #param_dists=param_dists[0] # pulls whole length 9 tuple out of the inputted version which has it all in one cell
   
    #params is a list of current values of parameters in same order as param_dists
    #Parameter distribution data needs to be read in using the following forms for the following distributions
    # binom, [c n,p]
    # triang, [current parameter value, c, loc, scale]
    # nbinom, [current parameteer value, n,p]


    # # this is for a normal calibration
    # freq_ES, time_recovery, advancing_to_symptoms, Rt, symptom_case_fatal, days_to_incubation, test_sensitivity, test_specificity,  new_infections_per_shock, initial_infect, delay, non_compliance  = param_dists  
    # this is for a normal calibration
    freq_ES, time_recovery, advancing_to_symptoms,  Rt, symptom_case_fatal, days_to_incubation, test_sensitivity, test_specificity,  new_infections_per_shock, initial_infect, initial_iso = param_dists  

    # Now we calculate our log prior using our prior distributions for our parameters
    
    # # below is for a normal calbration
    # log_prior_output = binom.logpmf(int(params[0]), freq_ES[0], freq_ES[1]) + triang.logpdf(float(params[1]), c = time_recovery[0],loc = time_recovery[1], scale = time_recovery[2]) + triang.logpdf(float(params[2]), c= advancing_to_symptoms[0], loc = advancing_to_symptoms[1], scale = advancing_to_symptoms[2]) + triang.logpdf(float(params[3]),  c= Rt[0], loc = Rt[1], scale = Rt[2]) + triang.logpdf(float(params[4]), c = symptom_case_fatal[0], loc = symptom_case_fatal[1], scale = symptom_case_fatal[2]) + triang.logpdf(float(params[5]), c = days_to_incubation[0], loc = days_to_incubation[1], scale = days_to_incubation[2]) + triang.logpdf(float(params[6]), c = test_sensitivity[0], loc = test_sensitivity[1], scale = test_sensitivity[2]) +  norm.logpdf(float(params[7]), loc = test_specificity[0], scale = test_specificity[1]) + nbinom.logpmf(int(params[8]), new_infections_per_shock[0],  new_infections_per_shock[1]) + nbinom.logpmf(int(params[9]), initial_infect[0], initial_infect[1]) + binom.logpmf(int(params[10]), delay[0], delay[1]) + truncnorm.logpdf(float(params[11]), non_compliance[0], non_compliance[1], loc = non_compliance[2], scale = non_compliance[3]) 
    
    # below is for a normal calbration
    log_prior_output = binom.logpmf(int(params[0]), freq_ES[0], freq_ES[1]) + triang.logpdf(float(params[1]), c = time_recovery[0],loc = time_recovery[1], scale = time_recovery[2]) + triang.logpdf(float(params[2]), c= advancing_to_symptoms[0], loc = advancing_to_symptoms[1], scale = advancing_to_symptoms[2]) + triang.logpdf(float(params[3]),  c= Rt[0], loc = Rt[1], scale = Rt[2])  + triang.logpdf(float(params[4]), c = symptom_case_fatal[0], loc = symptom_case_fatal[1], scale = symptom_case_fatal[2]) + triang.logpdf(float(params[5]), c = days_to_incubation[0], loc = days_to_incubation[1], scale = days_to_incubation[2]) + triang.logpdf(float(params[6]), c = test_sensitivity[0], loc = test_sensitivity[1], scale = test_sensitivity[2]) +  norm.logpdf(float(params[7]), loc = test_specificity[0], scale = test_specificity[1])  + nbinom.logpmf(int(params[8]), new_infections_per_shock[0],  new_infections_per_shock[1])  + nbinom.logpmf(int(params[9]), initial_infect[0], initial_infect[1]) + truncnorm.logpdf(params[10], initial_iso[0], initial_iso[1], loc = initial_iso[2], scale = initial_iso[3])
    
    
    return(log_prior_output) # returns log prior result
   
def log_likelihood_func(params, args):
   
    # Pulls needed items from args 
    settings = args[0]
    percent_subpop = args[1]
    vaccination_status = args[2]
    RIT_testing_data = args[3]
    N = args[4]
    Nsub = args[5]
    Nsim = args[6]
    input_settings = args[7]
    updated_positive_tests = args[8]
    weights_norm = args[9]

    
    # Initializations
    log_likelihood_output_per_day = np.zeros(int(N/3)) #initializes a vector of zeros to fill
    
    
    gamma_u = 1# Sets gamma for unvaccinated subpop to zero
    time_to_return_fps = 1 # sets timesteps to return false positive to one
    
    if gamma_u != 1:
        sys.exit('ERROR: Gamma parameter not equal to 1') #exits code if warning is triggered and prints why
    if time_to_return_fps !=1:
        sys.exit('ERROR: time to return FP parameter not equal to 1') #exits code if warning is triggered
    print('gamma', gamma_u)
    #below is for a normal calibration
    parameters = [params[0], params[1], params[2], params[3], params[4],  params[5], time_to_return_fps, params[6], params[7], gamma_u, int(params[9]), 0, 0, int(params[8]), params[10] ]
    print(parameters[14])
    parameters = np.resize(parameters,(15,1)) # resize it so it gets read in correctly
    
    gamma_v = np.ones(N)
    
    
    # new_infections_per_ES_input = int(10)*np.ones((N,Nsub)) # sets up new_infections input for each subpop at each timestep
    new_infections_per_ES_input = int(params[8])*np.ones((N,Nsub)) # sets up new_infections input for each subpop at each timestep

    # run the model    
    (tests_per_timestep, probability_positive_test, tests_per_day, I, IDivisor, new_infections_per_shock, parameters, 
     infection_tally, cumulative_infect_finalR, conservation_check,
     isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings,
                                                                           percent_subpop, new_infections_per_ES_input, 
                                                                           vaccination_status, gamma_v, RIT_testing_data, 
                                                                           Nsim) #run our model function
                                                                         

    
    # calculate the isolation pool per day and the difference between model pool and datat
    isolation_pool = np.zeros(int(N/3))
    isolation_diff = np.zeros(int(N/3))
    

    for k in range(0, int(N/3)):
        isolation_pool[k] = np.mean((isolation_pool_population[3*k], isolation_pool_population[(3*k)+1], isolation_pool_population[(3*k)+2])) #totals complete isolation pool for each day    
        isolation_diff[k] = abs(RIT_testing_data.iloc[k,4] - isolation_pool[k]) #Calculates difference between isolation pools for each day
   

    
    # Now we use the above to calculate our log_likelihood
    tests_per_day_in_data = RIT_testing_data.iloc[:,2] # number of tests per day from the RIT data
    num_positive_tests_in_data = updated_positive_tests# number of positive tests per day from the RIT data
    probability_positive_test_per_day = np.zeros(int(N/3)) # we have probability per timestep but we need probability per day so we initialize an emoty vector to fill
    U_A_pop_proportion_final = np.zeros(int(N/3))
    num_tests_U_A = np.zeros(int(N/3))
    iso_pop_data = RIT_testing_data.iloc[:,4]
    
    
   
    for i in range(0, int(N/3)-1):
        # print('MCMC', i, tests_per_day_in_data[i], probability_positive_test_per_day[i])
        # average three probabilities per timestep to get probability per day 
        probability_positive_test_per_day[0] = 0
        probability_positive_test_per_day[i+1] = np.mean((probability_positive_test[3 * i], probability_positive_test[(3 * i) + 1], probability_positive_test[(3 * i) + 2]))
        # print('mcmc', i+1, 3 * i, (3 * i) + 1, (3 * i) + 2)
        # # average three proportions per timestep 
        # U_A_pop_proportion_final[i] = np.mean((U_A_pop_proportion[3*i], U_A_pop_proportion[(3*i) + 1], U_A_pop_proportion[(3*i) + 2]))
        
        # # finds number of tests administered that day based on average proportion of pop in uninfected and asymptomatic
        # num_tests_U_A[i] = U_A_pop_proportion_final[i] * tests_per_day[i] 
        # if i == 0:
        #     print(0, 'U_A', U_A_pop_proportion_final[i+1], 'tests', tests_per_day[i], 'tests adapted', num_tests_U_A[i+1])
        # print(i+1, 'U_A', U_A_pop_proportion_final[i+1], 'tests', tests_per_day[i], 'tests adapted', num_tests_U_A[i+1])

        # Then we use the above values to find the log likelihood per day
        if iso_pop_data[i] == 0:
            if weights_norm[i] ==0:
                log_likelihood_output_per_day[i] = 0 + binom.logpmf( k = round(num_positive_tests_in_data[i]), n = round(tests_per_day_in_data[i]), p = probability_positive_test_per_day[i]) + norm.logpdf(x = isolation_pool[i], loc = iso_pop_data[i], scale =  np.sqrt(-np.log(0.025)/2))
            else:
                log_likelihood_output_per_day[i] =   np.log(weights_norm[i]) + binom.logpmf( k = round(num_positive_tests_in_data[i]), n = round(tests_per_day_in_data[i]), p = probability_positive_test_per_day[i]) + norm.logpdf(x = isolation_pool[i], loc = iso_pop_data[i], scale = np.sqrt(-np.log(0.025)/2))
        else:
            if weights_norm[i] ==0:
                log_likelihood_output_per_day[i] = 0 + binom.logpmf( k = round(num_positive_tests_in_data[i]), n = round(tests_per_day_in_data[i]), p = probability_positive_test_per_day[i]) +  norm.logpdf(x = isolation_pool[i], loc = iso_pop_data[i], scale = np.sqrt(iso_pop_data[i]))
            else:
                log_likelihood_output_per_day[i] =   np.log(weights_norm[i]) + binom.logpmf( k = round(num_positive_tests_in_data[i]), n = round(tests_per_day_in_data[i]), p = probability_positive_test_per_day[i]) +  norm.logpdf(x = isolation_pool[i], loc = iso_pop_data[i], scale = np.sqrt(iso_pop_data[i]))
        # print('mcmc', i,probability_positive_test_per_day[i] )
        # print('LL', i, log_likelihood_output_per_day[i], isolation_pool[i], iso_pop_data[i], norm.logpdf(x = isolation_pool[i], loc = iso_pop_data[i], scale = np.sqrt(iso_pop_data[i])))
        # print('w', i, weights_norm[i])
    #saving 
    probability_positive_test_per_day_df =pd.DataFrame(probability_positive_test_per_day)
    probability_positive_test_per_day_df.to_csv('/Users/meghanchilds/Desktop/mcmc_proba_check_rerun.csv')
    
    U_A_pop_proportion_final_df = pd.DataFrame(U_A_pop_proportion_final)
    U_A_pop_proportion_final_df.to_csv('/Users/meghanchilds/Desktop/U_A_prop_check_rerun.csv')
    
    num_tests_U_A_df = pd.DataFrame(num_tests_U_A)
    num_tests_U_A_df.to_csv('/Users/meghanchilds/Desktop/num_tests_U_A_check_rerun.csv')
    
    # Now we add our indepedent probabilities to get total log_likehood for the whole model run
    # normally we would multiply independent probabilities but these are the logs so we use addition per log rules
    # print('LL VEC', log_likelihood_output_per_day)
    log_likelihood_output = sum(log_likelihood_output_per_day)
    # print('ll output', log_likelihood_output)
    
    if input_settings.iloc[25,1] == 1: # this signifies the calibration benchmark is being run and we return 0 so that we are sampling from priors   
        return(0) # if calibration is run we return 0 to only sample from our priors not our posteriors
    else:
        return(log_likelihood_output)

def log_posterior_score(params, param_dists, args): #Now we use our log prior pdf to find our log posterior score
    
    
    log_prior_output = log_prior_pdf(params, param_dists)
    
    if log_prior_output == - np.inf or log_prior_output == np.inf: # checks if log_prior is -inf or inf if this is the case
    # then the log posterior score will always be -inf so we skip running the model as out of bounds params will cause errors
        log_likelihood_output = -np.inf # we set log liklihood output to -inf cuz it doesnt matter what it will be if log_prior output
        # is negative inf. This will also save us a model run if we do not need it as well as save us from triggering failsafes and cutting the calibration short
    
    else:
        log_likelihood_output = log_likelihood_func(params, args)
    
    log_posterior_score_output = log_prior_output + log_likelihood_output # adds the natural logs of our likelihood function to our log posterior score
    
    return(log_posterior_score_output)

def neg_log_posterior_score(params, related_info): # then since we are using logs we take the negative of the score
   
     # First we unpack the tuple called related info this contains 
    # settings, testing_data, percent_subpop, vaccination_status, RIT_Testing_data, N, Nsub, and param_dists
    settings = related_info[0]
    testing_data = related_info[1]
    percent_subpop = related_info[2]
    vaccination_status = related_info[3]
    RIT_testing_data = related_info[4]
    N = related_info[5]
    Nsub = related_info[6]
    param_dists = related_info[7]
    
    #Now we create the tuple that the log_likehood function needs
    args = (settings, testing_data, percent_subpop, vaccination_status, RIT_testing_data, N, Nsub)
   
    
    #Now we run the functions to get our log posterior score
    
    log_posterior_score_output = log_posterior_score(params, param_dists, args)
    
    # Then we multiply by negative 1 to get the negative log posterior score
    neg_log_posterior_score_output = -1 * log_posterior_score_output
    
    return(neg_log_posterior_score_output)


