#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:41:04 2023

@author: meghanchilds
"""

import numpy as np
import pandas as pd

# Metric of Interest Collection

def Metrics_Tally(N, Nsub, settings, vaccination_status, parameters, beta, theta, gamma, 
                  sigma, I, new_infections_per_shock, population_size, mat_subpop, 
                  mat_subpop_totals, gamma_v, test_sensitivity, 
                  test_specificity,RIT_testing_data):
    #Initialization#
    pos_test_results_ts = np.zeros((N,Nsub))  #Initializes matrix of zeros to fill
    false_pos_test_results_ts = np.zeros((N,Nsub))  #Initializes matrix of zeros to fill
    true_pos_test_results_ts = np.zeros((N,Nsub))  #Initializes matrix of zeros to fill
    # num_pos_tests_per_day = np.zeros(int(N/3))#Initializes vector of zeros to fill
    
    total_new_infections_per_2_days = np.zeros(int((N/3)/2))
    vaxxed_new_infections_per_2_days = np.zeros(int((N/3)/2))
    unvaxxed_new_infections_per_2_days = np.zeros(int((N/3)/2))
    total_new_infections_per_week= np.zeros(int((N/3)/7))
    vaxxed_new_infections_per_week = np.zeros(int((N/3)/7))
    unvaxxed_new_infections_per_week = np.zeros(int((N/3)/7))
    new_infections_per_timestep = np.zeros((N,Nsub))
    probability_positive_test = np.zeros(N)
    # probability_positive_test_adapt = np.zeros(N)
    # num_tests_adaptor = np.zeros(N)
    # probability_positive_test_PER_DAY = np.zeros(int(N/3)+1)
    # num_tests_PER_DAY_U_A = np.zeros(int(N/3)+1)
    # num_test_adaptor_PER_DAY = np.zeros(int(N/3)+1)

    
    # test_sensitivity = np.array([test_sensitivity])
    # test_sensitivity.reshape((Nsub, 1))
    # print(type(test_sensitivity))
    if settings[24] == 1:
       
        test_sensitivity = test_sensitivity[0] # these will be the same for both subpopulations so set them as a scalar since we are not summing over subpops but are using total values
        test_specificity = test_specificity[0]
        
   
    #Loops for different tallies and check because they do not need different first step#
    for i in range(0,N-1):
        
        if (i > 0 and settings[24] == 1) or (i > 0 and settings[32] == 1):
            # print('SUCCESS')

            
            probability_positive_test[i] = test_sensitivity *  (mat_subpop_totals[i,3] / ( mat_subpop_totals[i ,0] +  mat_subpop_totals[i,2] +  mat_subpop_totals[i,3])) + (1 - test_specificity) * ((mat_subpop_totals[i,0]) / ( mat_subpop_totals[i,0] +  mat_subpop_totals[i,2] +  mat_subpop_totals[i,3])) 
            # print('mcmc A', i, mat_subpop_totals[i,3])
            # print('proba', probability_positive_test[i])
            # print('mcmc', i, 'Se:', test_sensitivity, 'A:', mat_subpop_totals[i - 1,3], 'U:',  mat_subpop_totals[i-1 ,0], 'E:',  mat_subpop_totals[i-1,2], '1-Sp:', (1 - test_specificity) )
        for j in range(0,Nsub):
            # if i > 0: # only pull tests at 8 hours so cannot pull at first time step
                
            #     if cycles_per_test[i,j]==0: # if no tests occur then we adjust the values so the corresponding terms are zero so no individuals go into testing class
            #     # for when data is being used to calculate testing rate
            #         #print('Using zero test case') # for debugging
            #         testing_adjust_param=0 # makes whole term moving to testing zero since no tests are occuring
            #         cycles_adjust_param=1 # adds one to denominator to negate divide by zero error, does not matter since whole term will be multiplied by zero
            #     else:
            #         #print('Using nonzero test case') # for debugging
            #         testing_adjust_param=1 # both values do not change the value of the term because tests are not zero
            #         cycles_adjust_param=0
                
            # if i > 0 and settings[24] == 1: 
            #     probability_positive_test_adapt[i] = test_sensitivity *  (mat_subpop_totals[i-1,3] / ( mat_subpop_totals[i-1,0] +  mat_subpop_totals[i-1,3])) + (1 - test_specificity) * (mat_subpop_totals[i-1,0] / ( mat_subpop_totals[i-1,0] +  mat_subpop_totals[i-1,3])) 
            #     num_tests_adaptor[i] = (mat_subpop_totals[i-1,0] + mat_subpop_totals[i-1,3]) / ( mat_subpop_totals[i-1,0] +  mat_subpop_totals[i-1,2] +  mat_subpop_totals[i-1,3])
                  
            if int(vaccination_status[j]) == 1: #checks whether it is an unvaccinated or vaccinated subpopulation
               #print('VAXXED') # for deubbing
                   gamma=float(gamma_v[i])
                   # print('vaxxed', gamma)
                  
            else:
                   gamma=float(parameters[9,j]) #if it is an unvaccinated subpop we use the gamma parameter for unvaccinated individuals 
                   # print('unvaxxed', gamma)
            new_infections_per_timestep[i,j] = mat_subpop[i,2,j]*theta[j]#mat_subpop[i,0,j]*(beta[j]*(gamma)*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))) 
    probability_positive_test[0] = test_sensitivity *  (mat_subpop_totals[0,3] / ( mat_subpop_totals[0 ,0] +  mat_subpop_totals[0,2] +  mat_subpop_totals[0,3])) + (1 - test_specificity) * ((mat_subpop_totals[0,0]) / ( mat_subpop_totals[0,0] +  mat_subpop_totals[0,2] +  mat_subpop_totals[0,3])) 
    probability_positive_test_df = pd.DataFrame(probability_positive_test)
    probability_positive_test_df.to_csv(r'/Users/meghanchilds/Desktop/proba_mcmc_check.csv')
    
    
    # probability_positive_test_PER_DAY[0] = 0
    # num_test_adaptor_PER_DAY[0] = 0
    
    # for i in range(0, int(N/3)):
    #     probability_positive_test_PER_DAY[i] =  np.mean((probability_positive_test_adapt[3*i],  probability_positive_test_adapt[(3*i) + 1],  probability_positive_test_adapt[(3*i) + 2]))
    #     num_test_adaptor_PER_DAY[i] = num_tests_adaptor[3*i] + num_tests_adaptor[(3*i)+1] + num_tests_adaptor[(3*i)+2]
    #     num_tests_PER_DAY_U_A[i] = num_test_adaptor_PER_DAY[i] * RIT_testing_data.iloc[i,2]/3
    
    
    # if settings[32] == 1:
    #     # print('SUCCESS 2')
    #     probability_positive_test_per_day = np.zeros(int(N/3)) # we have probability per timestep but we need probability per day so we initialize an emoty vector to fill
       
    #     for i in range(0, int(N/3)-1):
            
    #         # average three probabilities per timestep to get probability per day 
    #         probability_positive_test_per_day[0] = 0
    #         probability_positive_test_per_day[i+1] = np.mean((probability_positive_test[3 * i], probability_positive_test[(3 * i) + 1], probability_positive_test[(3 * i) + 2]))
       
    #     df_proba_pos_bin_check  = pd.DataFrame(probability_positive_test_per_day)
        
    #     df_proba_pos_bin_check.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Hand Tuning Outputs/proba_pos_bin_check_recitaion_demo.csv',  header=False, mode = 'a')
    
    # num_tests_adaptor_df = pd.DataFrame(num_test_adaptor_PER_DAY)
    # num_tests_adaptor_df.to_csv(r'/Users/meghanchilds/Desktop/Fall_num_test_adaptor_debug_FPTP.csv',  header=False, index=False, mode = 'a')
    
    # num_tests_PER_DAY_U_A_df = pd.DataFrame(num_tests_PER_DAY_U_A)
    # num_tests_PER_DAY_U_A_df.to_csv(r'/Users/meghanchilds/Desktop/Fall_num_tests_debug_FPTP.csv',  header=False, index=False, mode = 'a')
    
    # proba_pos_test_results_ts_df = pd.DataFrame(probability_positive_test_PER_DAY)
    # proba_pos_test_results_ts_df.to_csv(r'/Users/meghanchilds/Desktop/Fall_proba_pos_test_per_day_debug_FPTP.csv',  header=False, index=False, mode = 'a')

    total_new_infection_timestep = np.sum(new_infections_per_timestep, axis = 1)
    # Seperate vaxxed and unvaxxed new infections per time step    
    if Nsub > 1:
        vaxxed_infections_per_timestep = new_infections_per_timestep[:,0]
        unvaxxed_infections_per_timestep = new_infections_per_timestep[:,1]  
    

    for i in range(int((N/3)/2)):
        total_new_infections_per_2_days[i] = sum(total_new_infection_timestep[6 * i : 6 * i + 6]) 
        if Nsub > 1:
            vaxxed_new_infections_per_2_days[i] = sum(vaxxed_infections_per_timestep[6 * i : 6 * i + 6])
            unvaxxed_new_infections_per_2_days[i] = sum(unvaxxed_infections_per_timestep[6 * i : 6 * i + 6])
   
    for i in range(int((N/3)/7)): 
        total_new_infections_per_week[i] = sum(total_new_infection_timestep[21 * i : 21 * i + 21]) 
        if Nsub > 1:
            vaxxed_new_infections_per_week[i] = sum(vaxxed_infections_per_timestep[21 * i : 21 * i + 21]) 
            unvaxxed_new_infections_per_week[i] = sum(unvaxxed_infections_per_timestep[21 * i : 21 * i + 21]) 
    
    #num_pos_test = np.sum(pos_test_results, axis = 1)  # sums number of positive tests across subpopulations 
    
    # # Now we comine the positive tests from all three time steps in a day
    # for m in range(int(N/3)):
    #     num_pos_tests_per_day[m] = np.sum(num_pos_test[3 * m : 3 * m + 3])

    max_total_new_infect_2_days = max(total_new_infections_per_2_days)
    max_total_new_infect_week = max(total_new_infections_per_week )
   
    if Nsub > 1:
        max_vaxxed_new_infect_2_days = max(vaxxed_new_infections_per_2_days)
        max_unvaxxed_new_infect_2_days = max(unvaxxed_new_infections_per_2_days)
        max_vaxxed_new_infect_week  = max(vaxxed_new_infections_per_week )
        max_unvaxxed_new_infect_week  = max(unvaxxed_new_infections_per_week )

    if Nsub > 1:
        return(probability_positive_test, total_new_infections_per_2_days, total_new_infections_per_week, 
           vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
           unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
           max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
           max_unvaxxed_new_infect_week, pos_test_results_ts, false_pos_test_results_ts, true_pos_test_results_ts   )
    else:
        return (probability_positive_test,  total_new_infections_per_2_days, total_new_infections_per_week, 
            max_total_new_infect_2_days,  max_total_new_infect_week, pos_test_results_ts, false_pos_test_results_ts, true_pos_test_results_ts  )
    
    