#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:18:00 2023

@author: meghanchilds
"""
import numpy as np
import pandas as pd
import math 

def Sobol_Data_Collection(input_settings, sim_num, param_values_test , max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, cumulative_sympto_infections_all_sims):
    
    Sobol_outputs=np.column_stack((sim_num, max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims, cumulative_sympto_infections_all_sims)) # combines out outputs of interest for MoM into a matrix
    
    df_Sobol_outputs=pd.DataFrame(Sobol_outputs, columns=['Number of Simulations', 'Max Isolation Population', 'Max Vaccinated Isolation Population', 'Max Unvaccinated Isolation Population', 'Cumulative Infections', 'Vaccinated Symptomatic Infections', 'Unvaccinated Symptomatic Infections', 'Cumulative Symptomatic Infections']) # combines matrix of outputs for MoM into a dataframe
    
    df_Sobol_outputs.to_csv(r'Sobol_Med_Risk_DVE_95v_5u_Only_Unvaxx_Testing_N_2_16_sims_Rerun_DELETE.csv', index=True)#, index=False) #exports the dataframe to a csv file on my desktop and appends the new inputs each time so we keep our prior data 
    
    if input_settings.iloc[19,1]==1: # onlt saves parameters when we note we are running a sobol sensitivity analysis
        
        Sobol_parameters=param_values_test    
        
        df_Sobol_parameters=pd.DataFrame(Sobol_parameters) # puts MoM parameters into a datframe
        
        df_Sobol_parameters.to_csv(r'Sobol_Params_N_2_16_DVE_Only_Unvaxx_Testing_Rerun_DELETE.csv', index=True) #exports the dataframe to a csv file on my desktop and appends the new inputs each time so we keep our prior data 

    return()

def RMSE_Calc_Data_Collection(simulation_number_vec, RMSE_vec):
   
    calibration_RMSE=np.column_stack((simulation_number_vec, RMSE_vec))
    
    df_calibration_RMSE=pd.DataFrame(calibration_RMSE, columns=['Simulation Number', 'RMSE Value'])    
    
    df_calibration_RMSE.to_csv('Calibration_RMSE.csv', index=False) #exports the dataframe to a csv file on my desktop and appends the new inputs each time so we keep our prior data 
    
    return()

def Random_Forest_Data_Collection(Nsim, Nsub, frequency_of_screening, frequency_ES, time_recovery, advancing_symptoms, 
                                  Rt, symptom_case_fatality_ratio, days_to_incubation, 
                                  time_to_return_fps, test_specificity, test_sensitivity,
                                  gamma_v_start, gamma_v_mod,  gamma_u, new_infections_per_shock_og,
                                  max_total_new_infect_2_days_all_sims, max_vaxxed_new_infect_2_days_all_sims,
                                  max_unvaxxed_new_infect_2_days_all_sims,max_total_new_infect_week_all_sims,
                                  max_vaxxed_new_infect_week_all_sims, max_unvaxxed_new_infect_week_all_sims, 
                                  max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, 
                                  cumulative_sympto_infections_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
                                  unvaxxed_sympto_infections_all_sims):
    if Nsub > 1:
        Model_Data_LHS = np.column_stack((frequency_of_screening[:,0], frequency_ES[:,0], time_recovery[:,0], advancing_symptoms[:,0], 
                                          Rt[:,0], symptom_case_fatality_ratio[:,0], days_to_incubation[:,0], 
                                          time_to_return_fps[:,0], test_specificity[:,0], test_sensitivity[:,0],
                                          gamma_v_start*np.ones(Nsim),  gamma_u*np.ones(Nsim), 
                                          new_infections_per_shock_og[:,0], max_total_new_infect_2_days_all_sims,
                                          max_vaxxed_new_infect_2_days_all_sims,max_unvaxxed_new_infect_2_days_all_sims,
                                          max_total_new_infect_week_all_sims, max_vaxxed_new_infect_week_all_sims,
                                          max_unvaxxed_new_infect_week_all_sims,max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, 
                                          max_unvaxxed_iso_population_all_sims, cumulative_sympto_infections_all_sims, 
                                          cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
                                          unvaxxed_sympto_infections_all_sims))
    
        Model_Data_LHS_df = pd.DataFrame(Model_Data_LHS, columns = ['frequency_of_screening',
                                                                    'frequency_ES', 
                                                                    'time_recovery', 
                                                                    'advancing_symptoms' ,
                                                                    'Rt', 
                                                                    'symptom_case_fatality_ratio', 
                                                                    'days_to_incubation', 
                                                                    'time_to_return_fps', 
                                                                    'test_specificity', 
                                                                    'test_sensitivity',
                                                                    'gamma_v_start', 
                                                                     
                                                                    'gamma_u',
                                                                    'new_infections_per_shock_og', 
                                                                    'max_total_new_infect_2_days_all_sims',
                                                                    'max_vaxxed_new_infect_2_days_all_sims',
                                                                    'max_unvaxxed_new_infect_2_days_all_sims',
                                                                    'max_total_new_infect_week_all_sims', 
                                                                    'max_vaxxed_new_infect_week_all_sims',
                                                                    'max_unvaxxed_new_infect_week_all_sims',
                                                                    'max_iso_population_all_sims', 
                                                                    'max_vaxxed_iso_population_all_sims', 
                                                                    'max_unvaxxed_iso_population_all_sims', 
                                                                    'cumulative_sympto_infections_all_sims', 
                                                                    'cumulative_infections_all_sims ',
                                                                    'vaxxed_sympto_infections_all_sims',
                                                                    'unvaxxed_sympto_infections_all_sims'])   
        
        Model_Data_LHS_df.to_csv(r'/Users/meghanchilds/Desktop/150000_Ensemble_for_Random_Forest_100unvaxx_med_risk_prior_Fall_2020_testing_data_no_calibration.csv')
    else:    
        Model_Data_LHS = np.column_stack((frequency_of_screening[:,0], frequency_ES[:,0], time_recovery[:,0], advancing_symptoms[:,0], 
                                          Rt[:,0], symptom_case_fatality_ratio[:,0], days_to_incubation[:,0], 
                                          time_to_return_fps[:,0], test_specificity[:,0], test_sensitivity[:,0],
                                          gamma_v_start[0]*np.ones(Nsim), gamma_u[0]*np.ones(Nsim), 
                                          new_infections_per_shock_og[:,0], max_total_new_infect_2_days_all_sims,
                                          max_total_new_infect_week_all_sims, max_iso_population_all_sims, 
                                          cumulative_sympto_infections_all_sims, cumulative_infections_all_sims))
    
        Model_Data_LHS_df = pd.DataFrame(Model_Data_LHS, columns = ['frequency_of_screening',
                                                                    'frequency_ES', 
                                                                    'time_recovery', 
                                                                    'advancing_symptoms' ,
                                                                    'Rt', 
                                                                    'symptom_case_fatality_ratio', 
                                                                    'days_to_incubation', 
                                                                    'time_to_return_fps', 
                                                                    'test_specificity', 
                                                                    'test_sensitivity',
                                                                    'gamma_v_start', 
                                                                    'gamma_u',
                                                                    'new_infections_per_shock_og', 
                                                                    'max_total_new_infect_2_days_all_sims',
                                                                    'max_total_new_infect_week_all_sims', 
                                                                    'max_iso_population_all_sims', 
                                                                    'cumulative_sympto_infections_all_sims', 
                                                                    'cumulative_infections_all_sims '
                                                                    ])   
        
        Model_Data_LHS_df.to_csv(r'/Users/meghanchilds/Desktop/150000_Ensemble_for_Random_Forest_100unvaxx_med_risk_prior_Fall_2020_testing_data_no_calibration.csv')
    
    return()

def Data_storage_paper_one_figures(max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, 
                                   cumulative_sympto_infections_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
                                   unvaxxed_sympto_infections_all_sims):
   
    Model_Data_LHS = np.column_stack((max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, 
    cumulative_sympto_infections_all_sims, cumulative_infections_all_sims, vaxxed_sympto_infections_all_sims,
    unvaxxed_sympto_infections_all_sims))
    
    Model_Data_LHS_df = pd.DataFrame(Model_Data_LHS, columns = ['max_iso_population_all_sims', 
                                                                'max_vaxxed_iso_population_all_sims', 
                                                                'max_unvaxxed_iso_population_all_sims', 
                                                                'cumulative_sympto_infections_all_sims', 
                                                                'cumulative_infections_all_sims ',
                                                                'vaxxed_sympto_infections_all_sims',
                                                                'unvaxxed_sympto_infections_all_sims'])   
    Model_Data_LHS_df.to_csv(r'/Users/meghanchilds/Desktop/95v_5u_No_Testing_Med_Risk_100000_DVE.csv')
    
    
    return( Model_Data_LHS_df)

def Metrics_of_Interest_Collection_Subpops(i, N, max_isolation_pop, max_vaxxed_isolation_pop, max_unvaxxed_isolation_pop, cumulative_infect_finalR, cumulative_symptomatic_infect_final_R, symptomatic_infection_tally,  max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims):
            print('COLLECTING METRICS OF INTEREST')
            max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
            max_vaxxed_iso_population_all_sims[i]=max_vaxxed_isolation_pop # Stores max vaxxed isolation population for each simulation
            max_unvaxxed_iso_population_all_sims[i]=max_unvaxxed_isolation_pop # Stores max unvaxxed isolation population for each simulation
            cumulative_infections_all_sims[i]=cumulative_infect_finalR # Stores cumulative infections for reach simulation
            cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulation
            vaxxed_sympto_infections=np.cumsum(symptomatic_infection_tally[:,0])
            unvaxxed_sympto_infections=np.cumsum(symptomatic_infection_tally[:,1])
            vaxxed_sympto_infections_all_sims[i]=vaxxed_sympto_infections[N-1]
            unvaxxed_sympto_infections_all_sims[i]=unvaxxed_sympto_infections[N-1]
            
            return(max_iso_population_all_sims, max_vaxxed_iso_population_all_sims, max_unvaxxed_iso_population_all_sims, cumulative_infections_all_sims, cumulative_sympto_infections_all_sims,vaxxed_sympto_infections_all_sims, unvaxxed_sympto_infections_all_sims)

def Metrics_of_Interest_Collection_Onepop(i, N, max_isolation_pop,cumulative_infect_finalR, cumulative_symptomatic_infect_final_R, max_iso_population_all_sims,cumulative_infections_all_sims, cumulative_sympto_infections_all_sims):
           
            max_iso_population_all_sims[i]=max_isolation_pop # Stores max total isolation population for each simulation
            cumulative_infections_all_sims[i]=cumulative_infect_finalR # Stores cumulative infections for reach simulation
            cumulative_sympto_infections_all_sims[i]=cumulative_symptomatic_infect_final_R # Stores cumulative symptomatic infections for reach simulatio
           
            return(max_iso_population_all_sims,cumulative_infections_all_sims, cumulative_sympto_infections_all_sims)
        
def Testing_Data_Collection( N, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, isolation_pool_population, RIT_testing_data, tests_administered_timestep, model_proba_pos_test):
    
    isolation_pool=np.zeros(int(N/3))#Initializes empty vector to fill
    
    
    pos_test_results_day = np.zeros(int(N/3))
    false_pos_test_results_day = np.zeros(int(N/3))
    true_pos_test_results_day = np.zeros(int(N/3))
    tests_administered_per_day = np.zeros(int(N/3))
    models_proba_pos_per_day = np.zeros(int(N/3))


    for k in range(0, int(N/3)):
        isolation_pool[k]=np.mean((isolation_pool_population[3* k], isolation_pool_population[(3 * k) + 1], isolation_pool_population[(3 * k) + 2])) #totals complete isolation pool for each day    
        models_proba_pos_per_day[k] = np.mean((model_proba_pos_test[3* k], model_proba_pos_test[(3 * k) + 1], model_proba_pos_test[(3 * k) + 2]))
        
        # # for deterministic testing term            
        # pos_test_results_day[k] =  num_pos_test_tally[3 * k] +  num_pos_test_tally[(3 * k) + 1] +  num_pos_test_tally[(3 * k) + 2]
        # false_pos_test_results_day[k] = false_pos_test_tally[3 * k] + false_pos_test_tally[(3 * k) + 1] + false_pos_test_tally[(3 * k) + 2]
        # true_pos_test_results_day[k] = true_pos_test_tally[3 * k] + true_pos_test_tally[(3 * k) + 1] + true_pos_test_tally[(3 * k) + 2]
        tests_administered_per_day[k] = tests_administered_timestep[3 * k] + tests_administered_timestep[(3 * k) + 1] + tests_administered_timestep[(3 * k) + 2]
        
        # for stochastic testing term
        false_pos_test_results_day[k] = false_pos_test_tally[k]
        true_pos_test_results_day[k] = true_pos_test_tally[k]
        pos_test_results_day[k] = num_pos_test_tally[k]
        
    # Saves data in correct format for model experiment
    # days = np.linspace(1, int(N/3), int(N/3))
    # model_experiment = np.column_stack((days, tests_administered_per_day, pos_test_results_day, isolation_pool, models_proba_pos_per_day))
    # model_experiment_df = pd.DataFrame(model_experiment, columns=['day', 'tests administered', 'pos_test_results_day', 'isolation_pool', 'models proba of a pos test'])
    # model_experiment_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Model_experiment_data_frame_rerun_APR_24.csv')
    
    
    #Now we calculate our Mean Squared Error and Root Mean Squared Error and make sure its ready to be put in the dataframe
    MSE=(np.square(np.subtract(isolation_pool, RIT_testing_data.iloc[:,4]))).mean() #calculates mean squared error
    RMSE=math.sqrt(MSE) #calculates root mean sqaured error
    #simulation_number_vec[i]=i #keeps track of simulation number so we know what RMSE pertains to what run number
    isolation_pool=np.transpose(isolation_pool)
    isolation_pool.resize((1, int(N/3))) # turns them into row vectors so each row of dataframe is a simulation
    
    pos_test_results_day = np.transpose(pos_test_results_day)
    pos_test_results_day.resize((1, int(N/3))) # turns them into row vectors so each row of dataframe is a simulation
    
    false_pos_test_results_day = np.transpose(false_pos_test_results_day)
    false_pos_test_results_day.resize((1, int(N/3))) # turns them into row vectors so each row of dataframe is a simulation
    
    true_pos_test_results_day = np.transpose(true_pos_test_results_day)
    true_pos_test_results_day.resize((1, int(N/3))) # turns them into row vectors so each row of dataframe is a simulation
    
    
    false_pos_test_results_day_df = pd.DataFrame( false_pos_test_results_day)
    false_pos_test_results_day_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Outputs from Model Runs with Calibrated Params/Fall_false_positive_test_per_day_df_all_params_FLF_100_sims_10_samples_rit_data_new_std_weighting_ll_multiple_4_sep_24.csv',  header=False, index=False, mode = 'a')
    
    true_pos_test_results_day_df = pd.DataFrame( true_pos_test_results_day)
    true_pos_test_results_day_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Outputs from Model Runs with Calibrated Params/Fall_true_positive_test_per_day_df_all_params_FLF_100_sims_10_samples_rit_data_new_std_weighting_ll_multiple_4_sep_24.csv',  header=False, index=False, mode = 'a')
    
    pos_test_results_day_df = pd.DataFrame( pos_test_results_day)
    pos_test_results_day_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Outputs from Model Runs with Calibrated Params/Fall_positive_test_per_day_df_all_params_FLF_100_sims_10_samples_rit_data_new_std_weighting_ll_multiple_4_sep_24.csv',  header=False, index=False, mode = 'a')
    
    isolation_pool_df = pd.DataFrame(isolation_pool)
    isolation_pool_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/Outputs from Model Runs with Calibrated Params/Fall_isolation_pool_df_all_params_FLF_100_sims_10_samples_rit_data_new_std_weighting_ll_multiple_4_sep_24.csv',  header=False, index=False, mode = 'a')
    
    
    return( isolation_pool, RMSE)