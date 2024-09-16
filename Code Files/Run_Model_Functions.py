## -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:35:42 2021
@author: admin
""" 


import numpy as np
import sys
import pandas as pd 
from Validation_Functions import validate_model
from Validation_Functions import run_benchmark_test
from Validation_Functions import run_VE_benchmark_test, run_testing_term_benchmark_test
from Metrics_tally_func import Metrics_Tally
from Param_and_Data_Organization_Funcs import Testing_Rate_calc, Total_Positive_Tests_per_Day, True_and_False_Positive
from Data_Collection_Scripts import Testing_Data_Collection
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform, truncnorm


def run_model(parameters, settings,  percent_subpop, new_infections_per_ES_input, vaccination_status, gamma_v, RIT_testing_data, Nsim):
   
    settings = settings
    ##Inputs##
    N = int(settings[1]) #Number of time steps
    Nsub = int(settings[2]) #Number of subpopulations
    population_size = settings[3]#Size of our total population
    cycles_per_day = settings[4] #8 hour time steps means 3 cycles per 24 hour period
    testing_data_input = settings[13]
    
    #If tetsing data is used lets split our daily tests among our timetsteps
    if testing_data_input == 1:
        tests_per_day = RIT_testing_data.iloc[:,2]
        
        tests_per_timestep = np.zeros(N)
        
        for j in range(0,N):
            tests_per_timestep[j] = tests_per_day[int(j/3)]/3
    # print(tests_per_day)  
    print(' new infect', new_infections_per_ES_input[0])
    ##Initialization##
    conservation_check=np.zeros(N) #Initializes vector of zeros to fill
    probability_positive_test_U_A = np.zeros(N) #Initializes vector of zeros to fill
    U_A_pop_proportion = np.zeros(N) #Initializes vector of zeros to fill
    new_infections_per_shock=np.zeros((N,Nsub)) #Initoializes matrix of zeros to fill
    false_pos_test_tally = np.zeros(int(N/3)) #Initializes vector of zeros to fill
    true_pos_test_tally = np.zeros(int(N/3)) #Initializes vector of zeros to fill
    num_pos_test_tally = np.zeros(int(N/3)) #Initializes vector of zeros to fill
    model_proba_pos_test = np.zeros(N) #Initializes vector of zeros to fill
    tests_administered_timestep = np.zeros(N) #Initializes vector of zeros to fill
    check_1=np.zeros(N) 
    check_2=np.zeros(N) 
    
    #Initlaize vectors for parameters
    frequency_ES=np.zeros(Nsub)
    rho=np.zeros(Nsub)
    sigma=np.zeros(Nsub)
    beta=np.zeros(Nsub)
    delta=np.zeros(Nsub)
    theta=np.zeros(Nsub)
    mu=np.zeros(Nsub)
    test_sensitivity=np.zeros(Nsub)
    test_specificity=np.zeros(Nsub)
    
    
    
    #Initialize our matrices for sub pops and total pops##
    mat_subpop=np.zeros((N,8,Nsub)) #matrix that holds all the subpopulations
    mat_subpop_totals=np.zeros((N,8)) #matrix that holds the total populations of all the compartments
    
    ##Initial Conditions##
    #Takes initial conditions from input_setiings CSV  file
    initial_false_positive = settings[6] #Initial False Positive Population
    # print('fp', initial_false_positive)
    initial_exposed = settings[7] #Initial Exposed Population
    # print('e', initial_exposed)
    initial_symptomatic = settings[8] #Initial Symptomatic Population
    # print('s', initial_symptomatic)
    if settings[31] == 1:
        initial_asymptomatic = parameters[10,0] #Initial Asymptomatic Population if initial infect is randomly sampled
        # init_infect_check = np.resize(np.array(initial_asymptomatic), (1,1))
        # df_init_infect = pd.DataFrame(init_infect_check )
        # df_init_infect.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/1st Run with Random Sample Initial Infect/MCMC_initial_infect_check.csv',  header=False, mode = 'a')
    else:
        initial_asymptomatic = settings[9] #Initial Asymptomatic Population if initiasl infect is not randomly sampled
    print('Initial Infected', initial_asymptomatic)
    if settings[39] == 1: # signifies adding uncertainity to iso pop
        initial_true_positive = parameters[14,:] # pulls in sampled param
    else: #otherwise uses the CSV value
        initial_true_positive = settings[10] #Initial True Positive Population
    print('tp', initial_true_positive)
    initial_recovered = settings[11] #Initial Recovered Population
    # print('r', initial_recovered)
    initial_dead = settings[12] #Initial Death Population
    # print('d', initial_dead)
    
    if settings[31] == 1:
        initial_uninfected = settings[3] - parameters[10,0] - initial_true_positive # if we randomly sample infeted that the number of uninfected is the total_pop minus the number of initially infected
    else: # otherwise we pull from the csv
        initial_uninfected = settings[5] #Initial Uninfected Population
    # print('u', initial_uninfected)
    ##Checks to make sure that initial compartment population inputs sum to equal total population input
    if (initial_uninfected+initial_false_positive+initial_exposed+initial_symptomatic+initial_asymptomatic+initial_true_positive+initial_recovered+initial_dead)!= population_size: 
        print(initial_uninfected+initial_false_positive+initial_exposed+initial_symptomatic+initial_asymptomatic+initial_true_positive+initial_recovered+initial_dead)

        print('ERROR:Populations do not equal total population!') #If populations do not match then it alerts us in the command window
        sys.exit() #exits code if warning is triggered
    
    ##Parameters##  
     
    # cycles_per_test=testing_data #if using testing data it is already in the form we need so we just assign it
    
    
    #Reads in values to calculate parameters or parameter values themselves
    frequency_ES_input=(parameters[0,:])#Exogenous shocks occur once a week so once here 21 time steps
    print('Freq ES', frequency_ES_input)
    time_recovery=(parameters[1,:]) #Takes _ days to recover
    print('time', time_recovery)
    advancing_to_symptoms=(parameters[2,:]) #percent who advance to symptoms
    # print('Advance', advancing_to_symptoms)
    Rt=(parameters[3,:]) #Rt value on average how many people is each infected person infecting
    print('Rt', Rt)
    symptom_case_fatality_ratio=(parameters[4,:]) #death rate
    # print('Fatal', symptom_case_fatality_ratio)
    days_to_incubation=(parameters[5,:])#days to incubation
    # print('days', days_to_incubation)
    time_to_return_fps=(parameters[6,:]) #days it takes to return to uninfectded from isolation after a false positive test
    delay = float(parameters[11,:]) # days before community transmission starts to take effect
    print('delay', delay)
    non_compliance = parameters[12,:] #percentage of people who are non-compliant with isolation protocol
    print('non-compliance', non_compliance)
    
    #Calculates parameters
    for i in range(0,Nsub):
        frequency_ES[i]=(cycles_per_day*float(frequency_ES_input[i])) #Frequency of Exogenous Shocks
        #print(float(time_recovery[i]))
        rho[i]=float(1/(float(time_recovery[i])*cycles_per_day)) #rate at which individuals recover from disease are removed
        sigma[i]=float((rho[i]*(float(advancing_to_symptoms[i])/100))/(1-float(advancing_to_symptoms[i])/100)) #rate of symptom onset for infected individuals
        beta[i]=float(float(Rt[i])*(rho[i]+sigma[i])) #rate at which infected individuals contact susceptibles and infect them
        delta[i]=float((float(symptom_case_fatality_ratio[i])/(1-float(symptom_case_fatality_ratio[i])))*rho[i]) #rate at which individuals in the symptomatic compartment die
        theta[i]=float(1/(float(days_to_incubation[i])*cycles_per_day)) #rate at which exposed individuals advance to the asymptomatic
        mu[i]=float(1/(cycles_per_day*float(time_to_return_fps[i]))) #rate at which false positives are returned to the unifected compartment
        if settings[34] == 1:
            test_specificity[i]=float(parameters[8,i]) #Specificity of COVID-test campuses are using ofr original model
        else:
            test_specificity[i] = np.sin(float(parameters[8,i])**2)
        
        print('SP', test_specificity, parameters[8,i])
        test_sensitivity[i]=float(parameters[7,i]) #Sensitivity of COVID-test campuses are using
        # print('SE', test_sensitivity)
        
    
    IDivisor=np.zeros(Nsub) #initializes empty vector to fill
    Adapted_IDivisor=np.zeros(Nsub) #initializes empty vector to fill
    I=np.zeros((N,Nsub));  #to create an indicator function which assumes value 1 if an exogenous shock takes place in cycle t; otherwise 0 
    # print('d', delay)
    # print('f',frequency_ES )
    
    
    
# =============================================================================
    ## FOR NON-STOCHASTIC FREQUENCY OF EXOGENOUS SHOCKS
# =============================================================================
    
    
    for k in range(0,Nsub):
        if frequency_ES[k] == 0: #if frequency of exogenous shocks is zero then I vector is all 0 because no exogenous shocks are occuring 
            I[:,k] = np.zeros(N);
            
        elif frequency_ES[k]== 3: #special case when frequency_ES_input is 1 and it will over reach in our index so we subtract one from our range   
                IDivisor[k]=(N)//(frequency_ES[k]) #Finds the integer divisor of N to to give us our range for I so it does not exceed the bounds
                Adapted_IDivisor[k]=(int(IDivisor[k])) #accounts for frewuency of ES input being 1 and subtracts 1 from range so we do not over count out index
                
                for i in range(1, int(Adapted_IDivisor[k])):
                        # Below if statement is for use if exogenous shocks are being delayed
                        if int(frequency_ES[k]) * i > delay * cycles_per_day: # sets threshold as delay parameter (in days) in terms of timesteps and sees if we are beyond that timestep
                            I[int(frequency_ES[k]) * i , k] = 1 #substututes 1 instead of 0 for every frequency of exogenous shock entry in I
                        else:
                            I[int(frequency_ES[k]) * i , k] = 0
                
        else:
                IDivisor[k] = (N)//(frequency_ES[k]) #Finds the integer divisor of N to to give us our range for I so it does not exceed the bound
                
                if IDivisor[k]*frequency_ES[k] >= N:
                    last_iter = int(IDivisor[k])
                else:
                    last_iter = int(IDivisor[k])+1

                for i in range(1, last_iter):
                
                    # Below if statement is for use if exogenous shocks are being delayed
                    if int(frequency_ES[k]) * i > delay * cycles_per_day: # sets threshold as delay parameter (in days) in terms of timesteps and sees if we are beyond that timestep
                        I[int( frequency_ES[k]) * i, k ] = 1 #substututes 1 instead of 0 for every frequency of exogenous shock entry in I
                    else:
                        I[int( frequency_ES[k]) * i, k ] = 0
    
    
    ##Inputting Initial Conditions##
    #Equally divides compartment population amoung subpopulations
    for j in range (0,Nsub):
        
        mat_subpop[0,0,j] = initial_uninfected*percent_subpop[j] #divides initial uninfected people equally among uninfected subpopulations
        # print(mat_subpop[0,0,j])
        
        mat_subpop[0,1,j] = initial_false_positive*percent_subpop[j] #divides initial false positive people equally among false positive subpopulations
        # print(mat_subpop[0,1,j])

        mat_subpop[0,2,j] = initial_exposed*percent_subpop[j] #divides initial exposed people equally among exposed subpopulations
        # print(mat_subpop[0,2,j])

        mat_subpop[0,3,j] = initial_asymptomatic*percent_subpop[j] #divides initial asymptomatic people equally among asymptomatic subpopulations
        # print(mat_subpop[0,3,j])

        mat_subpop[0,4,j] = initial_symptomatic*percent_subpop[j] #divides initial symptomatic people equally among symptomatic subpopulations
        # print(mat_subpop[0,4,j])

        mat_subpop[0,5,j] = initial_true_positive*percent_subpop[j] #divides initial true positive people equally among true positive subpopulations
        # print(mat_subpop[0,5,j])

        mat_subpop[0,6,j] = initial_recovered*percent_subpop[j] #divides initial recovered people equally among recovered subpopulations
        # print(mat_subpop[0,6,j])

        mat_subpop[0,7,j] = initial_dead*percent_subpop[j] #divides initial dead people equally among dead subpopulations
        # print(mat_subpop[0,7,j])

    ##Function Definitions for Updating Equations##

    #Function Defininition for first steps#
    def run_first_time_step(j, mat_subpop, mat_subpop_totals, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, gamma, gamma_deaths, non_compliance, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally):
        ##Uninfected Population##
       
        if gamma != 1:
            sys.exit('ERROR: Gamma parameter not equal to 1')
        if gamma_deaths != 1:
            sys.exit('ERROR: Gamma death parameter not equal to 1')
           
        (num_pos_tests_model, true_pos_test, false_pos_test, probability_positive_test_U_A, U_A_pop_proportion) = Total_Positive_Tests_per_Day(0, j, probability_positive_test_U_A, U_A_pop_proportion, 
                                                                                            test_sensitivity, test_specificity, mat_subpop_totals, RIT_testing_data, settings)
        # Tallies for stochastic testing term
        false_pos_test_tally[int(i/3)] = false_pos_test # stores false positve test results for each day
        # print(0, 'FP Tally', false_pos_test_tally[0])
        true_pos_test_tally[int(i/3)] = true_pos_test # stores true positve test results for each day
        # print(0, 'TP Tally', true_pos_test_tally[0])
        num_pos_test_tally[int(i/3)] = num_pos_tests_model
        # print(0, 'Pos tests Tally', num_pos_test_tally[0])

       #Uninfected Population
        mat_subpop_totals[1,0]=sum(mat_subpop[1,0,:])  #Totals uninfected subpopulations   
        mat_subpop[1,0,j]=mat_subpop[0,0,j]*(1-beta[j]*(gamma)*mat_subpop_totals[0,3]/(mat_subpop_totals[0,0]+mat_subpop_totals[0,3]+mat_subpop_totals[0,2]))+mat_subpop[0,1,j]*mu[j]  # first updating equation for our uninfected subpopulations  
        # print('gamma', gamma)
        # False Positive Population
        mat_subpop_totals[1,1]=sum(mat_subpop[1,1,:]) #Totals false positive subpopulations
        mat_subpop[1,1,j]=mat_subpop[0,1,j]*(1-mu[j]) # first updating equation for our false positive subpopulations
        # print('FP', 0, false_pos_test/3)
        
        # Exposed Population
        mat_subpop_totals[1,2]=sum(mat_subpop[1,2,:]) #Totals exposed subpopulations  
        mat_subpop[1,2,j]=mat_subpop[0,2,j]*(1-theta[j])+beta[j]*(gamma)*(mat_subpop[0,0,j]*mat_subpop_totals[0,3]/(mat_subpop_totals[0,0]+mat_subpop_totals[0,3]+mat_subpop_totals[0,2])) #first updating equation for our exposed subpopulations
        
        #Asymptomatic Population
        mat_subpop_totals[1,3]=sum(mat_subpop[1,3,:]) #Totals asymptomatic subpopulations
        mat_subpop[1,3,j]=mat_subpop[0,3,j]*(1-(gamma*sigma[j]*(1 - float(non_compliance[j])))-rho[j])+mat_subpop[0,2,j]*theta[j] #first updating equation for our asymptomatic subpopulations
        
        #Symptomatic Population
        mat_subpop_totals[1,4]=sum(mat_subpop[1,4,:]) #Totals symptomatic subpopulations
        mat_subpop[1,4,j]=mat_subpop[0,4,j]*(1-(gamma_deaths*delta[j])-rho[j])+((mat_subpop[0,3,j] * (1 - float(non_compliance[j]))) + mat_subpop[0,5,j])*(gamma*sigma[j]) #first updating equation for our symptomatic subpopulations
        
        # True Positive Population
        mat_subpop_totals[1,5]=sum(mat_subpop[1,5,:])#Totals true positive subpopulations
        mat_subpop[1,5,j]=mat_subpop[0,5,j]*(1-(gamma*sigma[j])-rho[j]) #first updating equation for our true positive subpopulations
        # print('TP', 0, true_pos_test/3)
        
        # Recovered Population
        mat_subpop_totals[1,6]=sum(mat_subpop[1,6,:]) #Totals recovered population
        mat_subpop[1,6,j]=mat_subpop[0,6,j]+(mat_subpop[0,3,j]+mat_subpop[0,4,j]+mat_subpop[0,5,j])*rho[j] #first updating equation for our recovered subpopulations
        
        #Deceased Population
        mat_subpop_totals[1,7]=sum(mat_subpop[1,7,:]) #Totals deceased population
        mat_subpop[1,7,j]=(gamma_deaths*delta[j])*mat_subpop[0,4,j]+mat_subpop[0,7,j] #first updating equation for our deceased subpopulations
        
        
        return(mat_subpop, mat_subpop_totals, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, false_pos_test, true_pos_test)
        
    # #Function Definition for rest of time steps#
    def run_time_step(i, j, mat_subpop, mat_subpop_totals, I, new_infections_per_shock, new_infections_per_ES_input, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, gamma, gamma_deaths, testing_data_input, RIT_testing_data, Nsim, non_compliance, tests_per_timestep, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, tests_administered_timestep, model_proba_pos_test, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test, true_pos_test, Nsub, N):
        
        if gamma != 1:
            sys.exit('ERROR: Gamma parameter not equal to 1')
        if gamma_deaths != 1:
            sys.exit('ERROR: Gamma death parameter not equal to 1') # warns if gamma parameter is wrong
        
        
                
        # for stochastic testing term
        if i % 3 == 0:
            
            (num_pos_tests_model, true_pos_test, false_pos_test, probability_positive_test_U_A, U_A_pop_proportion) = Total_Positive_Tests_per_Day(i, j, probability_positive_test_U_A, U_A_pop_proportion, test_sensitivity, test_specificity, 
                                                                                                mat_subpop_totals, RIT_testing_data, settings)
            false_pos_test_tally[int(i/3)] = false_pos_test # stores false positve test results for each day
            # print(i, 'FP Tally', false_pos_test_tally[int(i/3)])
            true_pos_test_tally[int(i/3)] = true_pos_test # stores true positve test results for each day
            # print(i, 'TP Tally', true_pos_test_tally[int(i/3)])
            num_pos_test_tally[int(i/3)] = num_pos_tests_model
            # print(i, 'Pos tests Tally', num_pos_test_tally[int(i/3)])
        
        if i < 3: # added so equations do not break during the first day when no testing is occuring
            # print('i < 3 case triggered')
            true_pos_test = 0
            false_pos_test = 0
            num_pos_tests_model = 0
            
        new_infections_per_shock_input=new_infections_per_ES_input
        # print('New infect', new_infections_per_shock_input)
        
        mat_subpop_totals[i,0]=sum(mat_subpop[i,0,:]) #Totals uninfected subpopulations
        mat_subpop_totals[i,2]=sum(mat_subpop[i,2,:]) #Totals exposed subpopulations
        mat_subpop_totals[i,3]=sum(mat_subpop[i,3,:]) #Totals asymptomatic subpopulations
        
        
        new_infections_per_shock[i,j] = min(float(new_infections_per_shock_input[j]), (mat_subpop[i,0,j]*(1-beta[j]*gamma*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2])))+mat_subpop[i,1,j]*mu[j]-false_pos_test/3)) #Checks to find min between our new infections per shock input and how many people remain in uninfected at that time step, prevents us from overpulling humans out of our uninfected compartment
        # if I[i+1,j] > 0:
        #     print('I',i, I[i+1,j])
        # print('inner check', new_infections_per_shock[i,j])
        
        #Uninfected Population
        mat_subpop[i+1,0,j]=mat_subpop[i,0,j]*(1-beta[j]*(gamma)*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2])))+mat_subpop[i,1,j]*mu[j]- false_pos_test/3 -I[i+1,j]*new_infections_per_shock[i,j] #updating equation for uninfected subpopulation, j, at time step i
        # print('gamma', gamma)
        # deterministic testing term
        #((1 - test_specificity[j]) * (1 - float(non_compliance[j])) * (mat_subpop_totals[i-1,0]/(mat_subpop_totals[i-1,0] + mat_subpop_totals[i-1,2] + mat_subpop_totals[i-1,3])) * tests_per_timestep[i])
        # print('FP', i, false_pos_test/3)
        # print('1-nc', (1 - float(non_compliance[j])))
        # print('gamma', gamma, beta[j]*(gamma)*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2])))
        # print('Infection proba at timestep i', i, (beta[j]*(gamma)*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))))
        ##False Positive Population##
        mat_subpop_totals[i,1]=sum(mat_subpop[i,1,:]) #Totals false positive subpopulations
        mat_subpop[i+1,1,j]=mat_subpop[i,1,j]*(1-mu[j]) + false_pos_test/3 #updating equation for false positive subpopulation, j, at time step i
        
        # deterministic testing term
        #((1 - test_specificity[j])*(1 - float(non_compliance[j])) * (mat_subpop_totals[i- 1,0]/(mat_subpop_totals[i- 1,0] + mat_subpop_totals[i- 1,2] +mat_subpop_totals[i- 1,3])) * tests_per_timestep[i])
        
        ##Exposed Population
        mat_subpop[i+1,2,j]=mat_subpop[i,2,j]*(1-theta[j])+beta[j]*(gamma)*(mat_subpop[i,0,j]*mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))+I[i+1,j]*new_infections_per_shock[i,j] #updating equation for exposed subpopulation, j, at time step i
        
        ##Asymptomatic Population##
        mat_subpop_totals[i,3]=sum(mat_subpop[i,3,:]) #Totals asymptomatic subpopulations
        mat_subpop[i+1,3,j]=mat_subpop[i,3,j]*(1-(gamma*sigma[j]*(1 - float(non_compliance[j])))-rho[j])+mat_subpop[i,2,j]*theta[j]- true_pos_test/3 #updating equation for asymptomatic subpopulation, j, at time step i
         
        # deterministic term
        # (test_sensitivity[j] * (1 - float(non_compliance[j])) * (mat_subpop_totals[i-1,3]/(mat_subpop_totals[i-1,0] + mat_subpop_totals[i-1,2] + mat_subpop_totals[i-1,3])) * tests_per_timestep[i])
        
        # print('TP', i, true_pos_test/3)

        ##Symptomatic Population##
        mat_subpop_totals[i,4]=sum(mat_subpop[i,4,:]) #Totals symptomatic subpopulations
        mat_subpop[i+1,4,j]=mat_subpop[i,4,j]*(1-rho[j]-(gamma_deaths*delta[j]))+((mat_subpop[i,3,j] * (1 - float(non_compliance[j]))) + mat_subpop[i,5,j])*(gamma*sigma[j]) #updating equation for symptomatic subpopulation, j, at time step i
        
        ##True Positive Population##
        mat_subpop_totals[i,5]=sum(mat_subpop[i,5,:]) #Totals true positive subpopulations
        mat_subpop[i+1,5,j]=mat_subpop[i,5,j]*(1-(gamma*sigma[j])-rho[j]) + true_pos_test/3#updating equation for true positive subpopulation, j, at time step i

        # deterministic term
        # (test_sensitivity[j] *(1 - float(non_compliance[j])) * (mat_subpop_totals[i- 1,3]/(mat_subpop_totals[i- 1,0] + mat_subpop_totals[i- 1,2] +mat_subpop_totals[i- 1,3])) * tests_per_timestep[i])

        ##Recovered Population##
        mat_subpop_totals[i,6]=sum(mat_subpop[i,6,:]) #Totals recovered population
        mat_subpop[i+1,6,j]=mat_subpop[i,6,j]+(mat_subpop[i,3,j]+mat_subpop[i,4,j]+mat_subpop[i,5,j])*rho[j] #updating equation for recovered subpopulation, j, at time step i
        
        ##Deceased Population##
        mat_subpop_totals[i,7]=sum(mat_subpop[i,7,:]) #Totals deceased population
        mat_subpop[i+1,7,j]=(gamma_deaths*delta[j])*mat_subpop[i,4,j]+mat_subpop[i,7,j] #updating equation for deceased subpopulation, j, at time step i
        
        # # for deterministic testing term
        # false_pos_test_tally[i] =  ((1 - test_specificity[j]) * (1 - float(non_compliance[j])) * (mat_subpop_totals[i-1,0]/(mat_subpop_totals[i-1,0] + mat_subpop_totals[i-1,2] + mat_subpop_totals[i-1,3])) * tests_per_timestep[i]) # stores false positve test results for each timestep
        # true_pos_test_tally[i] = (test_sensitivity[j] * (1 - float(non_compliance[j])) * (mat_subpop_totals[i-1,3]/(mat_subpop_totals[i-1,0] + mat_subpop_totals[i-1,2] + mat_subpop_totals[i-1,3])) * tests_per_timestep[i])# stores true positve test results for each timestep
        # num_pos_test_tally[i] = false_pos_test_tally[i] + true_pos_test_tally[i]
        
        model_proba_pos_test[i] = test_sensitivity *  (mat_subpop_totals[i - 1,3] / ( mat_subpop_totals[i-1 ,0] +  mat_subpop_totals[i-1,2] +  mat_subpop_totals[i-1,3])) + (1 - test_specificity) * ((mat_subpop_totals[i-1,0]) / ( mat_subpop_totals[i-1,0] +  mat_subpop_totals[i-1,2] +  mat_subpop_totals[i-1,3])) 
        tests_administered_timestep[i] = tests_per_timestep[i]
        
        false_pos_test = false_pos_test
        true_pos_test = true_pos_test
        
        # # for stochastic frequency of es
        # if I[i] == 1:
        #     print(i, I[i])
            
        #     print('j', j, 'infection proba',  1 - (beta[j]*mat_subpop_totals[i,0]*mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2])), 'before', (beta[j]*(mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))), beta[j], mat_subpop_totals[i,0], mat_subpop_totals[i,2], mat_subpop_totals[i,3] )
        #     proba = 1 - (beta[j]*mat_subpop_totals[i,0]*mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))
        #     if proba < 0:
        #         proba = 0
        #     print(binom.rvs(14, proba , size = 1))
        #     freq_es_new = 1
        #     print('freq_es_new', freq_es_new, 'infection proba', 1 - (beta[j]*(mat_subpop_totals[i,0]* mat_subpop_totals[i,3]/(mat_subpop_totals[i,0]+mat_subpop_totals[i,3]+mat_subpop_totals[i,2]))))

        #     for k in range(Nsub):
        #         print('i', i, 'new i', i+freq_es_new*3 )
        #         if int(i+freq_es_new*3) < N:
        #             I[int(i+freq_es_new*3), k] = 1
                
        # print(i, 'FP', false_pos_test_tally[i], 'TP', true_pos_test_tally[i], 'Pos tests', num_pos_test_tally[i])
        return(I, new_infections_per_shock, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, mat_subpop_totals,  true_pos_test_tally, false_pos_test_tally, num_pos_test_tally, tests_administered_timestep, model_proba_pos_test, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test, true_pos_test)
    
    ##Updating Equations##

    ##Sums subpopulations to get total populations of each compartment##
    #Can be used for a conservation check#
    mat_subpop_totals[0,0]=sum(mat_subpop[0,0,:]) #totals uninfected subpops at initial timestep
    mat_subpop_totals[0,1]=sum(mat_subpop[0,1,:]) #totals false positive subpops at initial timestep
    mat_subpop_totals[0,2]=sum(mat_subpop[0,2,:]) #totals exposed subpops at initial timestep
    mat_subpop_totals[0,3]=sum(mat_subpop[0,3,:]) #totals asymptomatic subpops at initial timestep
    mat_subpop_totals[0,4]=sum(mat_subpop[0,4,:]) #totals symptomatic subpops at initial timestep
    mat_subpop_totals[0,5]=sum(mat_subpop[0,5,:]) #totals true positive subpops at initial timestep
    mat_subpop_totals[0,6]=sum(mat_subpop[0,6,:]) #totals recovered subpops at initial timestep
    mat_subpop_totals[0,7]=sum(mat_subpop[0,7,:]) #totals deceased subpops at initial timestep

    print('gamma',parameters[9,j] )
    print('gamma_v', gamma_v[0])
    
    #First time step
    for j in range(0,Nsub): #When we have mutliple subpopulations we need to cycle through all of them so we use a loop to run the function for each sub population we have.
       
        if int(vaccination_status[j]) == 1: #checks whether it is an unvaccinated or vaccinated subpopulation
          
                gamma=float(gamma_v[0]) #if it is a vaccinated subpop we use the gamma parameter for vaccinated individuals
             
                gamma_deaths=float(gamma_v[0])
        else:
                gamma=float(parameters[9,j]) #if it is an unvaccinated subpop we use the gamma parameter for unvaccinated individuals 
               
                gamma_deaths=float(parameters[9,j]) 
           
        # print(gamma)
        
        
        (mat_subpop, mat_subpop_totals, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, false_pos_test, true_pos_test) = run_first_time_step(j, mat_subpop, mat_subpop_totals, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, gamma, gamma_deaths, non_compliance, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally) #runs function of updating loop for first time step           
        
        
    #Rest of the time steps#   
    for i in range(1,N-1):
         for j in range(0,Nsub):
            if int(vaccination_status[j]) == 1: #checks whether it is an unvaccinated or vaccinated subpopulation
                
                    gamma=float(gamma_v[i]) #if it is a vaccinated subpop we use the gamma parameter for vaccinated individuals
                    
                    gamma_deaths=float(gamma_v[0])
            else:
                    gamma=float(parameters[9,j]) #if it is an unvaccinated subpop we use the gamma parameter for unvaccinated individuals 
                    
                    gamma_deaths=float(parameters[9,j])
            
            # print('gamma', i, gamma)
            
            (I, new_infections_per_shock, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, mat_subpop_totals,  true_pos_test_tally, false_pos_test_tally, num_pos_test_tally, tests_administered_timestep, model_proba_pos_test, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test, true_pos_test)  = run_time_step(i, j, mat_subpop, mat_subpop_totals, I, new_infections_per_shock, new_infections_per_ES_input, rho, sigma ,beta, theta, delta, mu, test_sensitivity, test_specificity, gamma, gamma_deaths, testing_data_input, RIT_testing_data, Nsim, non_compliance, tests_per_timestep, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, tests_administered_timestep, model_proba_pos_test, probability_positive_test_U_A, U_A_pop_proportion, false_pos_test, true_pos_test, Nsub, N) #runs function of updating loops for the rest of the time steps     
             
    #Tallies totals for last time step 
    mat_subpop_totals[N-1,0]=sum(mat_subpop[N-1,0,:]) #totals uninfected subpops at initial timestep
    mat_subpop_totals[N-1,1]=sum(mat_subpop[N-1,1,:]) #totals false positive subpops at initial timestep
    mat_subpop_totals[N-1,2]=sum(mat_subpop[N-1,2,:]) #totals exposed subpops at initial timestep
    mat_subpop_totals[N-1,3]=sum(mat_subpop[N-1,3,:]) #totals asymptomatic subpops at initial timestep 
    mat_subpop_totals[N-1,4]=sum(mat_subpop[N-1,4,:]) #totals symptomatic subpops at initial timestep
    mat_subpop_totals[N-1,5]=sum(mat_subpop[N-1,5,:]) #totals true positive subpops at initial timestep
    mat_subpop_totals[N-1,6]=sum(mat_subpop[N-1,6,:]) #totals recovered subpops at initial timestep
    mat_subpop_totals[N-1,7]=sum(mat_subpop[N-1,7,:]) #totals deceased subpops at initial timestep
    
    # mat_subpop_totals_df = pd.DataFrame(mat_subpop_totals)
    # mat_subpop_totals_df.to_csv('matsubpop_totals_model_exper_MAP_params_rt_fixed_new_infect_fixed_time_recov_fixed.csv')
    
    # probability_positive_test_U_A_df = pd.DataFrame(probability_positive_test_U_A)
    # # print(probability_positive_test_U_A_df)
    # probability_positive_test_U_A_df.to_csv(r'/Users/meghanchilds/Desktop/stochastic_check.csv')
    if settings[32] == 1: # triggers when we are hand tuning so we can see soime diagnostic plots
    
        # num positive test plots
        
        plt.figure()
        days_F = np.linspace(1, int(N/3), int(N/3))
        plt.plot(days_F, num_pos_test_tally)
        plt.bar(days_F, RIT_testing_data.iloc[:,3])
        plt.legend('Modeled Positive Tests', 'RIT Data Positive Tests')
        plt.xlabel('Days')
        plt.ylabel('Number of Positive Test')
        plt.show()
        
        plt.figure()
        # plt.bar(days_F, tests_per_day, width =1, label = 'tests administered')
        plt.bar(days_F, false_pos_test_tally, width =1, label = 'FP', color = 'orange')
        plt.bar(days_F, true_pos_test_tally, width =1, label = 'TP', color = 'green')
        # plt.ylim([0,2])
        # plt.scatter(days_F, num_pos_test_tally, color='black', label = 'total positives')
        plt.xlabel('Days')
        plt.ylabel('Number of Tests')
        plt.legend()
        plt.show()
        
    #Now we tally our m, etrics of interest
    if Nsub > 1:
        (probability_positive_test, total_new_infections_per_2_days, total_new_infections_per_week, 
           vaxxed_new_infections_per_2_days, vaxxed_new_infections_per_week, unvaxxed_new_infections_per_2_days, 
           unvaxxed_new_infections_per_week,  max_total_new_infect_2_days, max_vaxxed_new_infect_2_days, 
           max_unvaxxed_new_infect_2_days, max_total_new_infect_week, max_vaxxed_new_infect_week, 
           max_unvaxxed_new_infect_week, pos_test_results_ts , false_pos_test_results_ts, true_pos_test_results_ts  ) = Metrics_Tally(N, Nsub, settings, vaccination_status, parameters, beta, 
                                                                                                                                       theta, gamma, sigma, I, new_infections_per_shock, 
                                                                                                                                       population_size, mat_subpop, mat_subpop_totals, gamma_v, 
                                                                                                                                       test_sensitivity, test_specificity,RIT_testing_data)
    else:
       #print('Metric Tally Triggered')
       (probability_positive_test,  total_new_infections_per_2_days, total_new_infections_per_week, 
           max_total_new_infect_2_days,  max_total_new_infect_week, pos_test_results_ts, false_pos_test_results_ts, true_pos_test_results_ts  ) = Metrics_Tally(N, Nsub, settings, vaccination_status, parameters, beta, 
                                                                                                                                       theta, gamma, sigma, I, new_infections_per_shock, 
                                                                                                                                       population_size, mat_subpop, mat_subpop_totals, gamma_v, 
                                                                                                                                       test_sensitivity, test_specificity,RIT_testing_data)
                                                                                    
    #Runs Validation function to check conservation of humans check and cumulative infections
    (max_isolation_pop, max_unvaxxed_isolation_pop, 
           max_vaxxed_isolation_pop, unvaccinated_isolation_pool_population, vaccinated_isolation_pool_population, 
           cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R, cumulative_asymptomatic_infections, 
           symptomatic_infection_tally, symptomatic_infection_tally_combined,cumulative_symptomatic_infections,
           cumulative_infections,infection_tally_combined, infection_tally, cumulative_infect_finalR, 
           conservation_check, isolation_pool_population) = validate_model(N, Nsub, vaccination_status, parameters, 
                                                                           beta, theta, gamma, sigma, I,
                                                                           new_infections_per_shock, population_size,
                                                                           mat_subpop, mat_subpop_totals, gamma_v, 
                                                                            test_sensitivity,
                                                                           test_specificity, non_compliance)#runs validation function
                                                                                                                
    if settings[14]==1: #checks switch in settings csv to see if benchmark test needs to be run
        (total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference) = run_benchmark_test(N, mat_subpop_totals, 
                                                                                                                                     isolation_pool_population, 
                                                                                                                                     cumulative_infect_finalR) #runs benchmark test to test against paltiel dashboard values
    
    if settings[15]==1: #checks switch in settings csv to see if benchmark test needs to be run
        (total_pop_difference_VE,cumulative_infections_difference_VE, isolation_pool_difference_VE, 
         total_pop_benchmark_VE,total_pop_difference_VE)=run_VE_benchmark_test(N, mat_subpop_totals, isolation_pool_population, 
                                                                               cumulative_infect_finalR) #runs benchmark test to test against paltiel dashboard values
    if settings[33] == 1: # checks to see if testing term benchmark is being run
        (total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference) = run_testing_term_benchmark_test(N, mat_subpop_totals, 
                                                                                                                                              isolation_pool_population, 
                                                                                                                                              cumulative_infect_finalR)
                                                                                        
    if settings[22] == 1 or settings[35] == 1: # checks if model is being run with calibrated samples or we are running a model experiment
        Testing_Data_Collection( N, false_pos_test_tally, true_pos_test_tally, num_pos_test_tally, isolation_pool_population, RIT_testing_data, tests_administered_timestep, model_proba_pos_test)

                                                                               
    if Nsub > 1 and settings[18] == 1 :  
        # this is the case where we have vaccinated and unvaccinated subpop and we want to collect our metrics of interest
        #print('This one second')                                                                 
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
    
    elif Nsub == 1 and settings[18] == 1: 
        # This is the case where we have one subpop and we want to collect our metrics of interest
        
        return( total_new_infections_per_2_days, total_new_infections_per_week, 
               max_total_new_infect_2_days, max_total_new_infect_week, max_isolation_pop,  
               cumulative_symptomatic_infect_final_R, cumulative_asymptomatic_infect_final_R,  
               cumulative_asymptomatic_infections,symptomatic_infection_tally, symptomatic_infection_tally_combined, 
               cumulative_symptomatic_infections,cumulative_infections,infection_tally_combined, I, IDivisor, 
               new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
               isolation_pool_population, mat_subpop, mat_subpop_totals)
    
    elif settings[14]==1:
       # outputs tailored to original benchmark 
        return(total_pop_difference, cumulative_infections_difference, isolation_pool_difference, I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
               cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals)

    elif settings[15]==1:
        # outputs tailored to VE benchmark
        return(total_pop_difference_VE, cumulative_infections_difference_VE, isolation_pool_difference_VE,total_pop_benchmark_VE,total_pop_difference_VE, 
               I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, 
               conservation_check, isolation_pool_population,  mat_subpop, mat_subpop_totals)
    
    elif settings[33] == 1:
        #outputs tailored to testing term benchmark
        return(total_pop_benchmark,total_pop_difference, cumulative_infections_difference, isolation_pool_difference, I, IDivisor, new_infections_per_shock, parameters, infection_tally, 
               cumulative_infect_finalR, conservation_check, isolation_pool_population, mat_subpop, mat_subpop_totals)
    
    elif settings[24] == 1: # indicates we are running a calibration
        # print('Runing this one')
        return (tests_per_timestep, probability_positive_test, tests_per_day, I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check,
                isolation_pool_population,  mat_subpop, mat_subpop_totals)
    
    else:
        
        # This is our baseline mode where we just output base outputs
        return(I, IDivisor, new_infections_per_shock, parameters, infection_tally, cumulative_infect_finalR, conservation_check, 
        isolation_pool_population,  mat_subpop, mat_subpop_totals)
