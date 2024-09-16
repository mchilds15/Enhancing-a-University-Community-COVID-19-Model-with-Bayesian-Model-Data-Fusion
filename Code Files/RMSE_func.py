#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:02:39 2022

@author: meghanchilds
"""
import numpy as np
from Run_Model_Functions import run_model
from Model_Setup_And_Run_Functions import Randomly_Generated_Setup_And_Run_Model_Function
from Model_Setup_And_Run_Functions import Sobol_Model_Setup_and_Run_Function
from Model_Setup_And_Run_Functions import CSV_Model_Setup_And_Run_Function
from Model_Setup_And_Run_Functions import RMSE_func
import math

def RMSE_Func_calc(params, args):
            
            #print(len(params))
            #print('params in RMSE func',params)
            settings=args[0]
            testing_data=args[1]
            percent_subpop=args[2]
            vaccination_status=args[3]
            RIT_testing_data=args[4]
            N=args[5]
            Nsub=args[6]
            Nsim = args[7]
            input_settings = args[8]
            
            # sampled_params_p1=int(params[0])
            # sampled_params_p2=params[1:6]
            # sampled_params_p3=params[6:8]
            frequency_es = 7
            new_infect_per_shock = 2
            symptom_case_fatal_ratio = 0.0005
            time_to_return_fps=1
            gamma_u=1
            
            parameters=[int(params[0]), params[1], params[2], params[3], params[4], params[5], time_to_return_fps, params[6], params[7], gamma_u,   params[8]]
            #parameters=[frequency_es, params[0], params[1], params[2], symptom_case_fatal_ratio, params[3], time_to_return_fps, params[4], params[5], gamma_u,   new_infect_per_shock]

            # parameter_p1=np.append(sampled_params_p1,sampled_params_p2)
            # parameter_p2=np.append(parameter_p1, time_to_return_fps)
            # parameter_p3=np.append(parameter_p2,  sampled_params_p3)
            # parameter_p4=np.append(parameter_p3, gamma_u)
            # parameters=np.append(parameter_p4, int(params[8]))
            # #print(parameters[0])
            parameters=np.resize(parameters,(11,1))
            
            gamma_v=np.ones(N)
            new_infections_per_ES_input=int( params[8])*np.ones((N,Nsub))
            #new_infections_per_ES_input=float(params[11])*np.ones((N,Nsub))
            
            (tests_per_timestep, probability_positive_test, tests_per_day, I, IDivisor,
             new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
             conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, 
                                                                                                       testing_data, percent_subpop, 
                                                                                                       new_infections_per_ES_input, 
                                                                                                       vaccination_status, gamma_v, 
                                                                                                       RIT_testing_data, Nsim) #run our model function
            print('Sim', 'complete')
           
            isolation_pool=np.zeros(int(N/3))
            isolation_diff=np.zeros(int(N/3))
                #Data Comparison
            
           
            for k in range(0, int(N/3)):
                isolation_pool[k]=np.mean((isolation_pool_population[3*k], isolation_pool_population[(3*k)+1], isolation_pool_population[(3*k)+2])) #totals complete isolation pool for each day    
                #print('run',k)
                isolation_diff[k]=abs(RIT_testing_data.iloc[k,4]-isolation_pool[k]) #Calculates difference between isolation pools for each day
                                            
            max_iso_diff=max(isolation_diff) #finds max difference between isolation pools over the semester
            min_iso_diff=min(isolation_diff) #finds min difference between isolation pools over the semester
            
            #Now we calculate our Mean Squared Error and Root Mean Squared Error and make sure its ready to be put in the dataframe
            MSE=(np.square(np.subtract(isolation_pool, RIT_testing_data.iloc[:,4]))).mean() #calculates mean squared error
            RMSE=math.sqrt(MSE) #calculates root mean sqaured error
            #simulation_number=i #keeps track of simulation number so we know what RMSE pertains to what run number
            RMSE_vec=RMSE #stores all the RMSE in a vector        
                #Now lets store this RMSE and corresponding Simulation number in an output CSV
               # simulation_number_vec=i*np.ones(len(params))
            #print('RMSE', params, RMSE_vec)   
            return(RMSE_vec)