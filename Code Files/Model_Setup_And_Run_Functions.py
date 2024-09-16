
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:03:51 2022

@author: meghanchilds
"""

import numpy as np
import sys
import pandas as pd 
from Run_Model_Functions import run_model
#from Model_Prep_and_Model_Run_Function import Model_Prep_and_Model_Run_Function
from datetime import datetime
import math 
import matplotlib.pyplot as plt
from pyDOE import *
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform
from scipy.interpolate import interp1d

def Benchmark_Setup_And_Run_Model_Function(i, Nsim, N, Nsub, benchmark_parameter_settings,input_settings, 
                                           frequency_ES, time_recovery, advancing_symptoms,  Rt, 
                                           symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, 
                                           test_sensitivity, test_specificity, gamma_u, 
                                           new_infections_per_shock_input,new_infections_per_shock_og, cycles_per_test_parameter, X, Y, gamma_v, percent_subpop, RIT_testing_data):
    

        testing_startTime = datetime.now()

        frequency_of_screening=benchmark_parameter_settings.iloc[1,1:Nsub+1]
        
        # If it is not empty then pulls parameter from that cell to use for frequency of screening

        print('Using testing frequency from benchmark CSV file')   

        for k in range(0, Nsub):
            cycles_per_test_parameter[:,k]=(float(frequency_of_screening[k])*float(input_settings.iloc[4,1])) #How many time steps between tests 

        print('testing',datetime.now() - testing_startTime) 

        #print('CYCLES TEST', cycles_per_test_parameter[0,:])

        parameters_startTime = datetime.now()  
        
        #Combines paramaters into a matrix and gives new_infections per shock its own vector/matrix (depends on how many subpopulations we have)

        print('Bencmark parameters')
        
        parameters=np.vstack((frequency_ES, time_recovery, advancing_symptoms, 
                              Rt, symptom_case_fatality_ratio, days_to_incubation, 
                              time_to_return_fps, test_sensitivity, test_specificity, 
                              gamma_u))
        
        new_infections_per_ES_input = new_infections_per_shock_og,
        print('New infections per exogenous shock ', new_infections_per_ES_input)

        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file

        vaccination_status=benchmark_parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv

        print('storing parameters',datetime.now() - parameters_startTime)

        testing_data=cycles_per_test_parameter#if we are we  set  the testing data as the values calculated using the RIT testing data values       
        print(X, Y)
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation

        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            
        model_startTime = datetime.now()
        
        (total_pop_difference, cumulative_infections_difference, isolation_pool_difference, I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
               cumulative_infect_finalR, conservation_check, isolation_pool_population, 
               mat_subpop, mat_subpop_totals)= run_model(parameters, settings,
                                                         percent_subpop, new_infections_per_ES_input, 
                                                         vaccination_status, gamma_v, RIT_testing_data, Nsim)       
        print('Model Run',datetime.now() - model_startTime) 
        print('Sim', i, 'complete')

        return(total_pop_difference, cumulative_infections_difference, isolation_pool_difference, I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
               cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals)

def VE_Benchmark_Setup_And_Run_Model_Function( i,  Nsim, N, Nsub, VE_Benchmark_parameter_settings,parameter_settings, input_settings, 
                                              percent_subpop,  new_infections_per_shock_input, new_infections_per_shock_og,
                                              frequency_ES, time_recovery, advancing_symptoms, Rt, 
                                              symptom_case_fatality_ratio, days_to_incubation,
                                              time_to_return_fps, test_sensitivity, test_specificity, 
                                              gamma_u, cycles_per_test_parameter, X, Y, gamma_v, RIT_testing_data):
    
        testing_startTime = datetime.now()
        frequency_of_screening=[10000,7]#VE_Benchmark_parameter_settings.iloc[12,1:Nsub+1]
   
        # If it is not empty then pulls parameter from that cell to use for frequency of screening
        print('Using testing frequency from VE benchmark CSV file')   
       
        for k in range(0, Nsub):
            cycles_per_test_parameter[:,k]=(float(frequency_of_screening[k])*float(input_settings.iloc[4,1])) #How many time steps between tests
        
        print('testing',datetime.now() - testing_startTime) 
        
        if Nsub > 1 and input_settings.iloc[15,1] != 1:
           
            print('Splitting exogenous shocks proportionally') # for debugging
            
            for n in range(0,Nsub):
                new_infections_per_shock_input[n]=new_infections_per_shock_og[i]*percent_subpop[n] #splits up exogenous shocks proportionally
        else:
            
            print('NOT splitting exogenous shocks proportionally') # for debugging
            
            new_infections_per_shock_input=new_infections_per_shock_og
            print('New Infections per exogeneous', new_infections_per_shock_input)
        parameters_startTime = datetime.now()  
        
        #Combines paramaters into a matrix and gives new_infections per shock its own vector/matrix (depends on how many subpopulations we have)
        print('VE Benchmark parameters')
        
        parameters=np.vstack((frequency_ES, time_recovery, advancing_symptoms, 
                              Rt, symptom_case_fatality_ratio, days_to_incubation, 
                              time_to_return_fps, test_sensitivity, test_specificity, 
                              gamma_u))
        
        new_infections_per_ES_input=np.transpose(new_infections_per_shock_input)
        
        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
        
        vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
        
        print('storing parameters',datetime.now() - parameters_startTime) 
        
        testing_data=cycles_per_test_parameter #if we are we  set  the testing data as the values calculated using the RIT testing data values       
        
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
        print(X, Y)
        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            
        model_startTime = datetime.now()
        
        (total_pop_difference_VE,cumulative_infections_difference_VE, isolation_pool_difference_VE,total_pop_benchmark_VE,total_pop_difference_VE, 
               I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
               conservation_check,isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings,  
                                                                                                        percent_subpop, new_infections_per_ES_input,
                                                                                                        vaccination_status, gamma_v, 
                                                                                                        RIT_testing_data, Nsim) #run our model function
        
        print('Model Run',datetime.now() - model_startTime) 
        print('Sim', i, 'complete')
        
        return(total_pop_difference_VE, cumulative_infections_difference_VE, isolation_pool_difference_VE,total_pop_benchmark_VE,total_pop_difference_VE, 
               I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
               conservation_check,isolation_pool_population, mat_subpop, mat_subpop_totals)

def Randomly_Generated_Setup_And_Run_Model_Function(i, N, Nsim, Nsub, 
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
                                                    max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, RIT_testing_data):
        
        testing_startTime = datetime.now() 
        
        
            
        for k in range(0, Nsub):
          #print('i=',i)
          cycles_per_test_parameter[:,k]=(frequency_of_screening[i,k]*float(input_settings.iloc[4,1])) #How many time steps between tests 
          
          if input_settings.iloc[30 , 1] == 1: # triggers when we are only testing the unvaccinated subpop
          
              cycles_per_test_parameter[:,0] =(100000 * float(input_settings.iloc[4,1])) 
          #print('testing',datetime.now() - testing_startTime) 
          #print('CYCLES TEST', cycles_per_test_parameter[0,:])
        if Nsub > 1 and input_settings.iloc[15,1] != 1:
            print('Splitting exogenous shocks proportionally') # for debugging
            for n in range(0,Nsub):
                new_infections_per_shock_input[n]=new_infections_per_shock_og[i]*percent_subpop[n] #splits up exogenous shocks proportionally
        else:
            print('NOT splitting exogenous shocks proportionally') # for debugging
            new_infections_per_shock_input = np.array(new_infections_per_shock_og[i])
            new_infections_per_shock_input  = new_infections_per_shock_input.reshape((1,1))
        parameters_startTime = datetime.now()  
            
        parameters_startTime = datetime.now()  
        #print('Random Parameters')
        
        parameters=np.vstack((frequency_ES[i,:], (time_recovery[i,:]), (advancing_symptoms[i,:]), (Rt[i,:]), (symptom_case_fatality_ratio[i,:]), (days_to_incubation[i,:]), (time_to_return_fps[i,:]), (test_sensitivity[i,:]), (test_specificity[i,:]),(gamma_u[i,:]), (initial_infect[i,:]))) 
        
        new_infections_per_ES_input=new_infections_per_shock_input
        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
        vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
        #print('storing parameters',datetime.now() - parameters_startTime) 
        testing_data=cycles_per_test_parameter #if we are we  set  the testing data as the values calculated using the RIT testing data values       
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            # if diminish_VE(m) > 1: #checks to see if it is greater than 1
            #     gamma_v[m]=1 # if it is greater than 1 then we set the value to 1 so our max value is 1
        model_startTime = datetime.now()
        
        if Nsub > 1 and input_settings.iloc[18,1] == 1 :  
            # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
            #print('This one third')                                                                  
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
                   isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, percent_subpop, 
                                                                                         new_infections_per_ES_input, vaccination_status, 
                                                                                         gamma_v, RIT_testing_data, Nsim) #run our model function        #print('Model Run',datetime.now() - model_startTime) 
        
        elif Nsub == 1 and input_settings.iloc[18,1] == 1: 
            # This is the case where we have one subpop and we want to collect our metrics of interest
            
            ( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings, 
                                                                                         percent_subpop, new_infections_per_ES_input, 
                                                                                     vaccination_status, gamma_v, RIT_testing_data, Nsim) #run our model function        
        
         
        
        
                 
        
        else:
            
            # This is our baseline mode where we just output base outputs
            (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
            isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings,
                                                                                  percent_subpop, new_infections_per_ES_input, 
                                                                              vaccination_status, gamma_v, RIT_testing_data, Nsim) #run our model function        #print('Model Run',datetime.now() - model_startTime) 
        print('Model Run',datetime.now() - model_startTime) 
        print('Sim', i, 'complete')
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
              
              max_total_new_infect_2_days_all_sims[i] = max_total_new_infect_2_days 
              max_vaxxed_new_infect_2_days_all_sims[i] = max_vaxxed_new_infect_2_days
              max_unvaxxed_new_infect_2_days_all_sims[i] = max_unvaxxed_new_infect_2_days
              max_total_new_infect_week_all_sims[i] = max_total_new_infect_week
              max_vaxxed_new_infect_week_all_sims[i] = max_vaxxed_new_infect_week
              max_unvaxxed_new_infect_week_all_sims[i] = max_unvaxxed_new_infect_week 
            
            else: 
                max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
                cumulative_infections_all_sims[i]=cumulative_infect_finalR # Stores cumulative infections for reach simulation
                cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulation
                deaths_all_sims[i]=mat_subpop_totals[N-1,7] # Stores total deaths for each simulation
                
                max_total_new_infect_2_days_all_sims[i] = max_total_new_infect_2_days 
                max_total_new_infect_week_all_sims[i] = max_total_new_infect_week
                
        #max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
        frequency_of_screening_output=frequency_of_screening
        
        if input_settings.iloc[21,1]==1: #checks to see if we are running in parallel
            
            return(unvaxxed_deaths_all_sims[i],vaxxed_deaths_all_sims[i], deaths_all_sims[i], 
                   max_iso_population_all_sims[i], max_vaxxed_iso_population_all_sims[i], 
                   max_unvaxxed_iso_population_all_sims[i],cumulative_infections_all_sims[i], 
                   cumulative_sympto_infections_all_sims[i], vaxxed_sympto_infections_all_sims[i], 
                   unvaxxed_sympto_infections_all_sims[i],max_total_new_infect_2_days_all_sims[i],
                   max_vaxxed_new_infect_2_days_all_sims[i],max_unvaxxed_new_infect_2_days_all_sims[i],
                   max_total_new_infect_week_all_sims[i],max_vaxxed_new_infect_week_all_sims[i],
                   max_unvaxxed_new_infect_week_all_sims[i], i) #max_isolation_pop, max_unvaxxed_isolation_pop, max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals))
       
        elif input_settings.iloc[18,1]==1 or input_settings.iloc[19,1]==1 : # if we are not running in parallel checks to see if we are collecting outputs for Sobol
            print('THIS CASE')
            if Nsub > 1:
                return( total_new_infections_per_2_days, total_new_infections_per_week, 
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
                   vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims) 
            else: 
                print('ANDTHIS CASE')
                return( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop, cumulative_symptomatic_infect_final_R, 
                   cumulative_asymptomatic_infect_final_R, cumulative_asymptomatic_infections,symptomatic_infection_tally, 
                   symptomatic_infection_tally_combined, cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined,
                   I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals, max_iso_population_all_sims, cumulative_infections_all_sims, 
                   cumulative_sympto_infections_all_sims)
            
                  
            
        
        else: # if nor doing either of the abvove just returns normal outputs
            
            return (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
             isolation_pool_population, mat_subpop, mat_subpop_totals)

def Sobol_Model_Setup_and_Run_Function(i, Nsim, N, Nsub, input_settings, parameter_settings, param_values_test, percent_subpop, time_to_return_fps, frequency_of_screening_fill, cycles_per_test_parameter, 
                                              new_infections_per_shock_og, new_infections_per_shock_input, gamma_u, gamma_v, X, Y, max_iso_population_all_sims, 
                                              max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,
                                              vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, deaths_all_sims, vaxxed_deaths_all_sims, unvaxxed_deaths_all_sims,RIT_testing_data):  
    #for i in range(0,Nsim):                                                                                                                                                                                                         
        testing_startTime = datetime.now()    
        #print('Using testing frequency from Sobol Parameter generation')       
        
        for j in range(0,Nsim):
            frequency_of_screening_fill[j,:]=param_values_test[j,9]
        
        for k in range(0, Nsub):
            
            cycles_per_test_parameter[:,k]=(frequency_of_screening_fill[i,k]*float(input_settings.iloc[4,1])) #How many time steps between tests 
            
            if input_settings.iloc[30 , 1] == 1: # triggers when we are only testing the unvaccinated subpop
            
                cycles_per_test_parameter[:,0] =(100000 * float(input_settings.iloc[4,1])) 
       
        #print('testing',datetime.now() - testing_startTime) 
        #print('CYCLES TEST', cycles_per_test_parameter)
        if Nsub > 1 and input_settings.iloc[15,1] != 1:
            print('Splitting exogenous shocks proportionally') # for debugging
            for n in range(0,Nsub):
                new_infections_per_shock_input[n]=new_infections_per_shock_og[i]*percent_subpop[n] #splits up exogenous shocks proportionally
        else:
            print('NOT splitting exogenous shocks proportionally') # for debugging
            new_infections_per_shock_input = np.array(new_infections_per_shock_og[i])
            new_infections_per_shock_input  = new_infections_per_shock_input.reshape((1,1))
        parameters_startTime = datetime.now()  
        #print('Random Parameters for Sobol')
        parameters=np.zeros((10, Nsub))
        for l in range(0,6):
                parameters[l,:]=param_values_test[i,l]
        parameters[6,:]=time_to_return_fps[1,:]
        for l in range(7,9):        
            parameters[l,:]=param_values_test[i,l-1]
        parameters[9,:]=gamma_u[0,:]
        
        new_infections_per_ES_input=new_infections_per_shock_input
        print(new_infections_per_ES_input)
        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
        vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
        #print('storing parameters',datetime.now() - parameters_startTime)
        testing_data=cycles_per_test_parameter #if we are we  set  the testing data as the values calculated using the RIT testing data values       
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            # if diminish_VE(m) > 1: #checks to see if it is greater than 1
            #     gamma_v[m]=1 # if it is greater than 1 then we set the value to 1 so our max value is 1
        
        
        model_startTime = datetime.now()
        
        if Nsub > 1 and input_settings.iloc[18,1] == 1 :  
            # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
                                                                             
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
                   isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings, 
                                                                                         percent_subpop, new_infections_per_ES_input, 
                                                                                         vaccination_status, gamma_v, 
                                                                                         RIT_testing_data, Nsim)#run our model function
        
        elif Nsub == 1 and input_settings.iloc[18,1] == 1: 
            # This is the case where we have one subpop and we want to collect our metrics of interest
            
            ( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals)= run_model(parameters, settings,  
                                                                                         percent_subpop, new_infections_per_ES_input, 
                                                                                         vaccination_status, gamma_v, 
                                                                                         RIT_testing_data, Nsim)#run our model function
        else:
            
            # This is our baseline mode where we just output base outputs
            (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
            isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings,
                                                                                  percent_subpop, new_infections_per_ES_input, 
                                                                                  vaccination_status, gamma_v, 
                                                                                  RIT_testing_data, Nsim)

        print('Model Run',datetime.now() - model_startTime) 
        print('Sim', i, 'complete')
# simulation_number_vec_timestep=i*np.ones(N)
# populations_RMSE=np.column_stack((simulation_number_vec_timestep, mat_subpop_totals))
# df_populations_RMSE=pd.DataFrame(populations_RMSE, columns=['Simulation Number','Uninfected', 'False Positive', 'Exposed', 'Asymptomatic', 'Symptomatic', 'True Positive', 'Recovered', 'Deceased'])
# df_populations_RMSE.to_csv('Population_RMSE.csv', index=False, mode='a')

        ##METHOD OF MORRIS AND SOBOL DATA COLLECTION
        if input_settings.iloc[17,1]==1: #signals we are collecting data for method of morris so triggers the following, this way we are not using time to run it when we do not need it
            # Storing Parameters for MoM
            MoM_parameters[i,:]=[frequency_ES[i,1], time_recovery[i,1], advancing_symptoms[i,1], Rt[i,1], symptom_case_fatality_ratio[i,1], days_to_incubation[i,1], test_sensitivity[i,1], test_specificity[i,1], gamma_v_start[i], new_infections_per_shock_og[i], frequency_of_screening[i,1]] #collects parameter values for each simulation in the form we need to run method of morris
            #Storing outputs of interest for MoM and other analysis
       
        if input_settings.iloc[17,1]==1 or input_settings.iloc[18,1]==1  :
            if Nsub > 1:
                
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
                cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulation
                deaths_all_sims[i]=mat_subpop_totals[N-1,7] # Stores total deaths for each simulation

        
        if input_settings.iloc[21,1]==1: #checks to see if we are running in parallel
            
            return(unvaxxed_deaths_all_sims[i],vaxxed_deaths_all_sims[i], deaths_all_sims[i], 
                   max_iso_population_all_sims[i], max_vaxxed_iso_population_all_sims[i], 
                   max_unvaxxed_iso_population_all_sims[i],cumulative_infections_all_sims[i], 
                   cumulative_sympto_infections_all_sims[i], vaxxed_sympto_infections_all_sims[i], 
                   unvaxxed_sympto_infections_all_sims[i], i) #max_isolation_pop, max_unvaxxed_isolation_pop, max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals))
        
        elif  input_settings.iloc[18,1]==1 : # if we are not running in parallel checks to see if we are collecting outputs for Sobol
            if Nsub  > 1: 
                return ( total_new_infections_per_2_days, total_new_infections_per_week, 
                       vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
                       unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
                       max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
                       max_unvaxxed_new_infect_week, max_isolation_pop, max_unvaxxed_isolation_pop, 
                       max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                       cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                       cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                       cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                       new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                       isolation_pool_population, mat_subpop, mat_subpop_totals)
            else:
                    return( total_new_infections_per_2_days, total_new_infections_per_week, 
                           max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                           cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                           cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                           cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                           new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                           isolation_pool_population, mat_subpop, mat_subpop_totals)
                
        else:       
                return(I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
                       conservation_check,isolation_pool_population, mat_subpop, mat_subpop_totals)

        
def CSV_Model_Setup_And_Run_Function(i, N, Nsub, parameter_settings, input_settings,cycles_per_test_parameter, new_infections_per_shock_input, new_infections_per_shock_og, percent_subpop,
                                frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_sensitivity, test_specificity, 
                                gamma_u, gamma_v, X, Y, RIT_testing_data, Nsim):                                                                                                                                                                     
        testing_startTime = datetime.now()
        # If it is not empty then pulls parameter from that cell to use for frequency of screening
        print('Using testing frequency from CSV file')   
        frequency_of_screening=parameter_settings.iloc[1,1:Nsub+1] #How often people are screened (in days)
        for k in range(0, Nsub):
            cycles_per_test_parameter[:,k]=(float(frequency_of_screening[k])*float(input_settings.iloc[4,1])) #How many time steps between tests
        print('testing',datetime.now() - testing_startTime) 
        print('CYCLES TEST', cycles_per_test_parameter[0,:])
        if Nsub > 1 and input_settings.iloc[15,1] != 1:
            print('Splitting exogenous shocks proportionally') # for debugging
            for n in range(0,Nsub):
                new_infections_per_shock_input[n]=new_infections_per_shock_og[i]*percent_subpop[n] #splits up exogenous shocks proportionally
        else:
            print('NOT splitting exogenous shocks proportionally') # for debugging
            new_infections_per_shock_input=new_infections_per_shock_og
        parameters_startTime = datetime.now()  
        print('CSV Parameters')    
        parameters=np.vstack((frequency_ES, time_recovery, advancing_symptoms, Rt, symptom_case_fatality_ratio, days_to_incubation, time_to_return_fps, test_sensitivity, test_specificity, gamma_u))
        new_infections_per_ES_input=new_infections_per_shock_input
        settings=input_settings.iloc[:,1] #selects setting values from the settings csv file
        vaccination_status=parameter_settings.iloc[14,1:Nsub+1] #pulls vaccination status of each subpopulation from parameter csv
        print('storing parameters',datetime.now() - parameters_startTime) 
        testing_data=cycles_per_test_parameter  #if we are we  set  the testing data as the values calculated using the RIT testing data values       
        diminish_VE=interp1d(X,Y[:,i]) # defines the interpolated line between gamma_v_start and gamma_v_end for each simulation
        for m in range (0,N):
            gamma_v[m]=diminish_VE(m) # defines a gamma value for each time step
            # if diminish_VE(m) > 1: #checks to see if it is greater than 1
            #     gamma_v[m]=1 # if it is greater than 1 then we set the value to 1 so our max value is 1
        model_startTime = datetime.now()
        
        if Nsub > 1 and input_settings.iloc[18,1] == 1 :  
            # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
                                                                             
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
                  isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings,  
                                                                                        percent_subpop, new_infections_per_ES_input, 
                                                                                        vaccination_status, gamma_v, RIT_testing_data, Nsim) #run our model function
        
        elif Nsub == 1 and input_settings.iloc[18,1] == 1: 
            # This is the case where we have one subpop and we want to collect our metrics of interest
            
            ( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, 
                                                                                         percent_subpop, new_infections_per_ES_input, 
                                                                                         vaccination_status, gamma_v, RIT_testing_data, Nsim) #run our model function
        else:
            
            # This is our baseline mode where we just output base outputs
            (I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
            isolation_pool_population, mat_subpop, mat_subpop_totals) = run_model(parameters, settings, 
                                                                                  percent_subpop, new_infections_per_ES_input, 
                                                                                  vaccination_status, gamma_v, RIT_testing_data, Nsim) #run our model function
        
        print('Model Run',datetime.now() - model_startTime) 
        print('Sim', i, 'complete')
        
        if Nsub > 1 and input_settings.iloc[18,1] == 1 :  
            # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
                                                                          
            return( total_new_infections_per_2_days, total_new_infections_per_week, 
                   vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
                   unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
                   max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
                   max_unvaxxed_new_infect_week, max_isolation_pop, max_unvaxxed_isolation_pop, 
                   max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals)
        
        elif Nsub == 1 and input_settings.iloc[18,1] == 1: 
            # This is the case where we have one subpop and we want to collect our metrics of interest
            
            return( total_new_infections_per_2_days, total_new_infections_per_week, 
                   max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
                   cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
                   cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
                   cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
                   new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
                   isolation_pool_population, mat_subpop, mat_subpop_totals)
        else:
            
            # This is our baseline mode where we just output base outputs
            return(I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
            isolation_pool_population, mat_subpop, mat_subpop_totals)
        
def RMSE_func(isolation_pool, RIT_testing_data, i):
    squared_error=np.square(np.subtract(isolation_pool, RIT_testing_data.iloc[:,4]))
    mean_squared_error=np.mean(squared_error)
    root_mean_squared_error=math.sqrt(mean_squared_error)
    RMSE_vec[i]=root_mean_squared_error
    return(RMSE_vec)