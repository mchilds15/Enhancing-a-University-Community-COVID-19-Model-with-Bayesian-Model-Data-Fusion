#  -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 05:24:44 2021
@author: admin
"""

"""
Created on Mon Jul 19 12:39:52 2021
@author: admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##Validation tests##

def validate_model(N, Nsub, vaccination_status, parameters, beta, theta, gamma, sigma, I, new_infections_per_shock, population_size, 
                   mat_subpop, mat_subpop_totals, gamma_v, test_sensitivity, test_specificity, non_compliance):
    #Initialization#
    cumulative_infections=np.zeros(N)#Initializes vector of zeros to fill
    infection_tally=np.zeros((N,Nsub)) #Initializes vector of zeros to fill
    infection_tally_combined=np.zeros(N) #Initializes vector of zeros to fill
    symptomatic_infection_tally=np.zeros((N,Nsub)) #Initializes vector of zeros to fill
    symptomatic_infection_tally_combined=np.zeros(N)#Initializes vector of zeros to fill
    conservation_check=np.zeros(N) #Initializes vector of zeros to fill
    isolation_pool_population=np.zeros((N))#Initializes matrix of zeros to fill
    vaccinated_isolation_pool_population=np.zeros((N))#Initializes matrix of zeros to fill
    unvaccinated_isolation_pool_population=np.zeros((N))#Initializes matrix of zeros to fill
    pos_test_results = np.zeros((N,Nsub))  #Initializes matrix of zeros to fill
    num_pos_tests_per_day = np.zeros(int(N/3))#Initializes vector of zeros to fill
    
    ##Conservation of humans check, Infections Count, Isolation Pool Check##
    #Loops for different tallies and check because they do not need different first step#
    for i in range(0,N-1):
        conservation_check[i]=mat_subpop_totals[i,0]+mat_subpop_totals[i,1]+mat_subpop_totals[i,2]+mat_subpop_totals[i,3]+mat_subpop_totals[i,4]+mat_subpop_totals[i,5]+mat_subpop_totals[i,6]+mat_subpop_totals[i,7] #checks to make sure humans are conserved
        for j in range(0,Nsub):
            isolation_pool_population[i]=mat_subpop_totals[i,1]+mat_subpop_totals[i,4]+mat_subpop_totals[i,5] #checks sixe of isolation pool each time step to check against Paltiel Dashboard
            
            if int(vaccination_status[j]) == 1: #checks whether it is an unvaccinated or vaccinated subpopulation
                #print('VAXXED') # for deubbing
                    gamma=float(gamma_v[i])
                    vaccinated_isolation_pool_population[i]=mat_subpop[i,1,j]+mat_subpop[i,4,j]+mat_subpop[i,5,j] #tallies only vaccinated populations contribution to isolation population
                    #print(gamma)
            else:
                    gamma=float(parameters[9,j]) #if it is an unvaccinated subpop we use the gamma parameter for unvaccinated individuals 
                    unvaccinated_isolation_pool_population[i]=mat_subpop[i,1,j]+mat_subpop[i,4,j]+mat_subpop[i,5,j] #tallies only unvaccinated populations contribution to isolation population
            
            

            infection_tally[i,j]=gamma*beta[j]*(mat_subpop[i,0,j]*mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))+I[i+1,j]*new_infections_per_shock[i,j]#totals toal number of infections each time step
           
            infection_tally_combined[i]=sum(infection_tally[i,0:Nsub]) #finds total across subpopulations, sums rows
            
            cumulative_infections=np.cumsum(infection_tally_combined) #finds the cumulative sum of the new infections
            
            
            symptomatic_infection_tally[i,j]=((mat_subpop[i,3,j] * (1 - float(non_compliance[j]))) + mat_subpop[i,5,j])*(gamma*sigma[j])
            symptomatic_infection_tally_combined[i]=sum(symptomatic_infection_tally[i,0:Nsub])
            cumulative_symptomatic_infections=np.cumsum(symptomatic_infection_tally_combined)
            
    
    cumulative_infection_final=cumulative_infections[N-1]  #Isolates last cumulative infection entry which is total cumulative infection                                     
    
    cumulative_infect_finalR=(cumulative_infection_final)#round(cumulative_infection_final)  #Rounds cumulative infections to get non decimal infection count 
    cumulative_symptomatic_infect_final_R=cumulative_symptomatic_infections[N-1]
    
    cumulative_asymptomatic_infections=cumulative_infections-cumulative_symptomatic_infections #defines cumulative asymptomatic infections
    cumulative_asymptomatic_infect_final_R=cumulative_asymptomatic_infections[N-1]#round(cumulative_asymptomatic_infections[N-1])
    
    
    isolation_pool_population[N-1]=mat_subpop_totals[N-1,1]+mat_subpop_totals[N-1,4]+mat_subpop_totals[N-1,5]
    
    if Nsub == 2:
        symptomatic_infection_tally[N-1,0]=((mat_subpop[N-1,3,0] * (1 - float(non_compliance[j])) )+mat_subpop[N-1,5,0])*(float(gamma_v[i])*sigma[0]) #defines symptomatic tally at last timestep for both subpops
        symptomatic_infection_tally[N-1,1]=((mat_subpop[N-1,3,1] * (1 - float(non_compliance[j])) )+mat_subpop[N-1,5,1])*(float(parameters[9,j]) *sigma[1])
        vaccinated_isolation_pool_population[N-1]=mat_subpop[N-1,1,0]+mat_subpop[N-1,4,0]+mat_subpop[N-1,5,0] #tallies only vaccinated populations contribution to isolation population
        unvaccinated_isolation_pool_population[N-1]=mat_subpop[N-1,1,1]+mat_subpop[N-1,4,1]+mat_subpop[N-1,5,1] #tallies only unvaccinated populations contribution to isolation population
    max_vaxxed_isolation_pop=max(vaccinated_isolation_pool_population) # finds max vaccinated isolation population for this simulation
    max_unvaxxed_isolation_pop=max(unvaccinated_isolation_pool_population) # finds max unvaccinated isolation population for this simulation
    max_isolation_pop=max(isolation_pool_population) #finds max isolation population for this simulation
    
    #print('iso_pool',isolation_pool_population[N-1])
    #Checks to make sure conservation of humans hold 
    if abs(conservation_check[i]-population_size)>10**-10:
        print("ERROR: Model is not conserving humans at time step", i, conservation_check[i]-population_size,population_size, conservation_check[i]) #If our model does not conserve humans it prints and error in the command window and alerts us to where this divergence occurs
    
    return(max_isolation_pop, max_unvaxxed_isolation_pop, 
           max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
           cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R, cumulative_asymptomatic_infections, 
           symptomatic_infection_tally, symptomatic_infection_tally_combined,cumulative_symptomatic_infections,
           cumulative_infections,infection_tally_combined, infection_tally, cumulative_infect_finalR, conservation_check, isolation_pool_population)
 
        


def run_benchmark_test(N, mat_subpop_totals, isolation_pool_population, cumulative_infect_finalR):
    
    #First we read in benchmark CSV
    benchmark_data=pd.read_csv(r'/Users/meghanchilds/Desktop/CSVs to run Model/Benchmark CSVs/Benchmark_Test_Values.csv') #reads in our saved benchmark data to test against from a csv
   
    
    #Second lets set up the data we need
    cumulative_infections_vec=cumulative_infect_finalR*np.ones(N) #makes model's cumulative infections into a vector to match dataframe setup
    
    total_pop_benchmark=np.array(benchmark_data.iloc[:,0:8]) #isolates the columns thta contain the total population matrix from the benchmark data
    cumulative_infections_benchmark=benchmark_data.iloc[:,9] #isolates the column that holds the cumulative infections from the benchmark data
    isolation_pool_benchmark=benchmark_data.iloc[:,8] #isolates the column that holds the isolation pool population from the benchmark data
    
    #Third we compare our values
    total_pop_difference=np.array(abs(mat_subpop_totals-total_pop_benchmark)) #calculates the difference between our models total populations and the benchmark total populations
    isolation_pool_difference=abs(isolation_pool_population-isolation_pool_benchmark) #calculates the difference between the model's isolation pool population and the benchmark isolation pool population
    cumulative_infections_difference=abs(cumulative_infections_vec-cumulative_infections_benchmark) #calculates the difference between the model's cumulative infections and the benchmark cumulative infections
   
    ##Fourth we check each difference matrix and vector component by component and warn in the command window if they do not match
    
    for i in range(0,N-1):
        for j in range(0,8):
            if abs(total_pop_difference[i,j]) > 10**-8: #checks total population differences
                print('Total populations do not match at', '(',i, ',', j,')') #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
                #break # for debugging
        if isolation_pool_difference[i] > 10**-8: #checks isolation pool population differences
            print('Isolation pool populations do not match at timestep', i) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging 
        if cumulative_infections_difference[i] > 10**-8: #checks cumulative infections differences
            print('Cumulative Infections do not match at timestep', i) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging 
    print('Benchmark Test Completed')
    return (total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference)

def run_VE_benchmark_test(N, mat_subpop_totals, isolation_pool_population, cumulative_infect_finalR):
    
    #First we read in benchmark CSV
    VE_benchmark_data=pd.read_csv(r'/Users/meghanchilds/Desktop/CSVs to run Model/Benchmark CSVs/VE_Benchmark_Test_Values.csv') #reads in our saved benchmark data to test against from a csv
   
    
    #Second lets set up the data we need
    cumulative_infections_vec=cumulative_infect_finalR*np.ones(N) #makes model's cumulative infections into a vector to match dataframe setup
    
    total_pop_benchmark_VE=VE_benchmark_data.iloc[:,0:8] #isolates the columns thta contain the total population matrix from the benchmark data
    cumulative_infections_benchmark_VE=VE_benchmark_data.iloc[:,9] #isolates the column that holds the cumulative infections from the benchmark data
    isolation_pool_benchmark_VE=VE_benchmark_data.iloc[:,8] #isolates the column that holds the isolation pool population from the benchmark data
    
    #Third we compare our values
    total_pop_difference_VE=abs(mat_subpop_totals-total_pop_benchmark_VE) #calculates the difference between our models total populations and the benchmark total populations
    isolation_pool_difference_VE=abs(isolation_pool_population-isolation_pool_benchmark_VE) #calculates the difference between the model's isolation pool population and the benchmark isolation pool population
    cumulative_infections_difference_VE=abs(cumulative_infections_vec-cumulative_infections_benchmark_VE) #calculates the difference between the model's cumulative infections and the benchmark cumulative infections
   
    ##Fourth we check each difference matrix and vector component by component and warn in the command window if they do not match
    
    for i in range(0,N-1):
        for j in range(0,8):
            if abs(total_pop_difference_VE.iloc[i,j]) > 10**-6: #checks total population differences
                print('Total populations do not match at', '(',i, ',', j,')', total_pop_difference_VE.iloc[i,j]) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
                #break # for debugging
        if isolation_pool_difference_VE[i] > 10**-6: #checks isolation pool population differences
            print('Isolation pool populations do not match at timestep', i, isolation_pool_difference_VE[i] ) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging
        if cumulative_infections_difference_VE[i] > 10**-6: #checks cumulative infections differences
            print('Cumulative Infections do not match at timestep', i) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging
    print('Benchmark Test Completed')
    return (total_pop_difference_VE, cumulative_infections_difference_VE, isolation_pool_difference_VE, total_pop_benchmark_VE,total_pop_difference_VE)


def run_testing_term_benchmark_test(N, mat_subpop_totals, isolation_pool_population, cumulative_infect_finalR):
    
    #First we read in benchmark CSV
    testing_term_benchmark_data = pd.read_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/CSVs To Run Model/Benchmark Testing/New_Testing_Term_Calibration_Values.csv') #reads in our saved benchmark data to test against from a csv
    del testing_term_benchmark_data['Unnamed: 0'] # deletes column added when its read in
    
    #Second lets set up the data we need
    cumulative_infections_vec = round(cumulative_infect_finalR)*np.ones(N) #makes model's cumulative infections into a vector to match dataframe setup
    
    total_pop_benchmark = np.array(testing_term_benchmark_data.iloc[:,0:8]) #isolates the columns thta contain the total population matrix from the benchmark data
    cumulative_infections_benchmark = testing_term_benchmark_data.iloc[:,9] #isolates the column that holds the cumulative infections from the benchmark data
    isolation_pool_benchmark = testing_term_benchmark_data.iloc[:,8] #isolates the column that holds the isolation pool population from the benchmark data
    
    #Third we compare our values
    total_pop_difference = np.array(abs(mat_subpop_totals-total_pop_benchmark)) #calculates the difference between our models total populations and the benchmark total populations
    isolation_pool_difference = abs(isolation_pool_population-isolation_pool_benchmark) #calculates the difference between the model's isolation pool population and the benchmark isolation pool population
    cumulative_infections_difference = abs(cumulative_infections_vec-cumulative_infections_benchmark) #calculates the difference between the model's cumulative infections and the benchmark cumulative infections
   
    ##Fourth we check each difference matrix and vector component by component and warn in the command window if they do not match
    
    for i in range(0,N-1):
        for j in range(0,8):
            if abs(total_pop_difference[i,j]) > 10**-8: #checks total population differences
                print('Total populations do not match at', '(',i, ',', j,')') #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
                #break # for debugging
        if isolation_pool_difference[i] > 10**-8: #checks isolation pool population differences
            print('Isolation pool populations do not match at timestep', i) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging 
        if cumulative_infections_difference[i] > 10**-8: #checks cumulative infections differences
            print('Cumulative Infections do not match at timestep', i) #if they do not match it prints a wanring in the command window and the indices of where the non-matching occurs
            #break # for debugging 
    print('Benchmark Test Completed')
    return (total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference)