#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:32:05 2022

@author: meghanchilds
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt 
import seaborn as sns
from RMSE_func import RMSE_Func_calc
from log_prior_functions import log_prior_pdf
from log_prior_functions import log_posterior_score
from scipy.stats.distributions import norm, binom, nbinom, triang, uniform
from progress.bar import IncrementalBar
# from chainconsumer import ChainConsumer
from tqdm import tqdm, trange
import math

def MCMC(num_iter, step_size, initial_parameters, param_dists, args, rand_seed_input, adaption_switch, covar_input):
    
    
    if rand_seed_input != -1: 
        # print('Running version')
        rand_seed = rand_seed_input
        np.random.seed(rand_seed_input)
        
    else:
        rand_seed = np.random.randint(0, 1000, size = 1)
        np.random.seed(rand_seed)
    # Initializations
    num_params = len(initial_parameters) if hasattr(initial_parameters,"__len__") else 1 # number of parameters
    parameters = np.zeros((num_iter, num_params)) # creates empty vector to store chosen parameter values for each parameter at each iteration
    num_success = 0 # initialize number of sucesses to start at zeo

    # i is for indexing parameter number 0-8
    ## param order is freq of ES, time recovery, advancing to symptoms, Rt, symtom case fatal, days to incubation, test sens, test spec, new infections per shock
    # t is for iteration number 0-num_iter
    
    # Store Initial Values
    parameters[0,:] = initial_parameters
    
    # Then we calculate our initial log posterior score
    log_posteriors = [log_posterior_score(parameters[0,:], param_dists, args)]
    
    # Now we run through the rest of the iterations
    # bar = IncrementalBar('MCMC', max = num_iter)
    for t in trange(num_iter - 1):
        #propose  a new set of params/state
        
        if adaption_switch == False: # if we are inputting a covariance matrix to restart a change this is the covariance matrix for adaptive metro hastings to read in
            previous_covar = covar_input
        elif adaption_switch == True and t == 0:
            previous_covar = np.zeros((num_params, num_params)) # sets an empty filler for when we have not defined previous covariance yet
        
        covar = adaptive_metro_hastings(step_size, parameters, num_params, t, previous_covar, adaption = adaption_switch)
        #proposal = stats.norm.rvs(loc = parameters[t,:], scale = step_size)
        proposal = stats.multivariate_normal.rvs(mean = parameters[t,:], cov = covar)
        previous_covar = covar # renames covariance to previous for use in adaptive metropolis hastings
        
        # randomly round binomial params (freq es, new infect, init infect) to avoid sitting on the boundary
        coin_flip = np.random.randint(0,2)
        # print('before', coin_flip, proposal[0], proposal[8], proposal[9])
        if coin_flip == 1:
            # rounds up
        
            proposal[0] = math.ceil(proposal[0]) # freq es
            proposal[8] = math.ceil(proposal[8]) # new infect
            proposal[9] = math.ceil(proposal[9]) # init infect
        if coin_flip == 0:
            # rounds down
            proposal[0] = math.floor(proposal[0]) # freq es
            proposal[8] = math.floor(proposal[8]) # new infect
            proposal[9] = math.floor(proposal[9]) # init infect
        
        # print('after', proposal[0], proposal[8], proposal[9])
        print(proposal)

        # Calculatelog_posterior score of the proposal
        
        proposal_log_posterior_score = log_posterior_score(proposal, param_dists, args)
        # print('proposed log post',proposal_log_posterior_score)
        # Calculate the acceptance probability

        p_accept = proposal_log_posterior_score - log_posteriors[-1]
        p_accept = np.min([p_accept, 0])
        
        #print('p_accept',p_accept) 
        
        rand = stats.uniform.rvs()
        #print(rand, np.exp(p_accept))
        # bar.next() 
        # Accept proposed parameters with probability p_accept
        if rand < np.exp(p_accept): # expoential to undo natural log since we are working with log prior and log posterior
               
            # Then we accept the proposed parameter 
            # print('Passes')
            
            # Store them in the parameters matrix
            #parameters = np.column_stack((parameters, proposal))
            parameters[t+1,:] = proposal
            
            # Add corresponding posterior score to end of list of posterior scores
            log_posteriors.append(proposal_log_posterior_score)
            
            
            # And update the number of successes
            num_success = num_success + 1
            
            #Periodically saving to a CSV
            if t % (num_iter/10) == 0 and t > 0:
                
                df_params=pd.DataFrame(parameters[int(t-(num_iter/10)):t,:], )
                df_log_posteriors=pd.DataFrame(log_posteriors[int(t-(num_iter/10)):t])
                df_params.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Chain_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_'  + str(rand_seed) + '.csv',  header=False, mode = 'a')
                df_log_posteriors.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Log_Post_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_' + str(rand_seed) + '.csv',  header=False, mode = 'a')
           
                       
        # Reject with probability 1-p_accept
        else:
            # We reject the proposed parameter 
            # print('Fails')
            
            # Add another copy of the current parameters value to the end of the parameter matrix
            parameters[t+1,:]=parameters[t,:]
            
            # Add corresponding log posterior score to the end of that list too
            
            log_posteriors.append(log_posteriors[-1]) 
            #log_posteriors[t] = (log_posterior_score(parameters[:, -1], param_dists, args))

            #log_posteriors.append(log_posterior_score(parameters[:,-1], param_dists, args))
            
            #Periodically saving to a CSV
            if t % (num_iter/10) == 0 and t > 0:
               
                df_params=pd.DataFrame(parameters[int(t-(num_iter/10)):t,:], )
                df_log_posteriors=pd.DataFrame(log_posteriors[int(t-(num_iter/10)):t])
                df_params.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Chain_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_' + str(rand_seed) + '.csv', header=False, mode = 'a')
                df_log_posteriors.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Log_Post_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_' + str(rand_seed) + '.csv',  header=False, mode = 'a')
        # print('log post:', log_posteriors[t]) 
        # print('params:', parameters[t,:])
        if t == num_iter - 2: # checks if we are on last interation
            print('saving covar', t)
            covar_df = pd.DataFrame(covar) # if yes saves the covariance matrix to a dataframe
            covar_df.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/covar_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_'+ str(rand_seed) + '.csv') # and exports covariance dataframe to a csv
                 
    df_params=pd.DataFrame(parameters[int((num_iter)-(num_iter/10)):int(num_iter),:],)
    df_log_posteriors=pd.DataFrame(log_posteriors[int((num_iter)-(num_iter/10)):int(num_iter)])
    df_params.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Chain_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_' + str(rand_seed) + '.csv',  header=False,  mode = 'a')
    df_log_posteriors.to_csv(r'/Users/meghanchilds/Desktop/Dissertation_Research/COVID-19-Model-Formal-Calibration/MCMC CSV Files/MCMC_Log_Post_Fall_Data_All_Params_100_iters_rit_data_stochastic_new_ll_adapted_binom_initial_iso_fixed_new_weighting_4_times_'  + str(rand_seed) + '.csv',  header=False,  mode = 'a')    
    
    
        
    
    return(parameters, log_posteriors, num_success, rand_seed, covar)

def adaptive_metro_hastings(step_size, parameters, num_params, t, previous_covar, adaption ):
    
    # step_size is initial covariance matrix that we choose
    # parameters is the parameter values we have calculated thus far in the chain
    # d is number of dimensions which in our case is the number of parameters
    # s_d is a parameter  that only depends on the dimension of the vectors
    # epsilon (eps) is a constant > 0 that we hoose to be very small compared to size of S
    # t is current iteration
   
    
    s_d = (2.4*2.4)/num_params # 10 is number of parameters
    eps = 1e-8 # epsilon value
    I_d = np.eye((num_params))# identity mstrix with dxd dimensions
    
    if adaption == False: #adapting is turned off
        covar = previous_covar # use given covariance matrix
        
    elif t < 1000: # if under adapting threshold use initial covariance matrix
        print(t, 'initial covar')
        
        covar=np.diag(step_size * step_size) # sets initial covarance matrix for initial timestep
        
    elif t >= 1000 and t%100==0: # if after adapting threshold and on an adapting iteration use adatptive metropolis hastings
        print(t, 'adapting')
        #print(parameters)
        #print(type(parameters),type(parameters[1]))

        #covar = (step_size**2) # sets initial covarance matrix for initial timestep
        # print('!!adapting covar!!')
        covar = s_d * np.cov(np.transpose(parameters[:t,:])) + s_d * eps * I_d
        # print(covar)
        #print(np.size(np.cov(parameters)))
    
    else: # if passed adapting threshold but not actively adapting use previous covariance
        print(t, 'previous')
        
        #print(np.size(np.cov((parameters))))
        covar = previous_covar # uses adapted covariance but does not adapt again yet
        
    # print(covar)
       #  
    return(covar)

def calibration_benchmark_test_plots(mcmc_output_0, mcmc_output_1, convergence_threshold,  Num_Params):
    #This function plots the posteriors against the priors to ensure when the benchmark test is run they are the same
    
    # chains = [ChainConsumer() for i in range(len(mcmc_output_1[0,:]))]
    # converged = [False for i in range(len(mcmc_output_1[0,:]))]
    # end = len(mcmc_output_1[:,0])
    # for i in range(0, len(mcmc_output_1[0,:])):
    #     # if i == 0:  
    #     #     name = 'Percent_advance' 
    #     # elif i == 1: 
    #     #     name = 'Days_to_Incubation'  
    #     # elif i == 2: 
    #     #     name = 'Test_Sensitivity'
    #     # elif i == 3: 
    #     #     name = 'Test_Specificity' 
       
    #     if i==0: 
    #         name = 'Freq_ES' 
        
    #     elif i==1:  
    #         name = "Time_Recov" 
    #     elif i==2: 
    #         name = 'Percent_advance' 
    #     elif i==3: 
    #         name = 'Rt' 
    #     elif i==4: 
    #         name = 'Symptom_fatal' 
    #     elif i==5: 
    #         name = 'Days_to_Incubation'  
    #     elif i==6: 
    #         name = 'Test_Sensitivity'
    #     elif i==7: 
    #         name = 'Test_Specificity' 
    #     elif i==8: 
    #         name ='New_Infect'
            
    #     chains[i].add_chain(np.concatenate((mcmc_output_0[:convergence_threshold,i], mcmc_output_1[:convergence_threshold,i])), walkers=2, name=name)
    #     converged = chains[i].diagnostic.gelman_rubin()
        # Now we overlay prior distributions on new posterior distributions
        # fig, axs = plt.subplots(2,1, sharex = False, sharey = False)
    
    # fig.set_figheight(10)
    # fig.set_figwidth(25)
    
    # # plt.subplot(3, 3, 1)
    # sns.kdeplot(mcmc_output_params[:,0], fill = True)
    # plt.plot(np.arange(0,15), binom.pmf(np.arange(0,15),14,0.5), color='k')
    # plt.axvline(x=7, color = 'r')
    # plt.legend(['Posterior Distribution', 'Prior Distribution'], fontsize = 16, loc = 'upper left')
    # # plt.xlabel('')
    # plt.ylabel('Density', fontsize=16)
    # plt.xticks(fontsize = 16)
    # plt.yticks([])
    # plt.xlabel('Frequency of Exogenous Shocks (days)', fontsize = 20)
    
    # plt.subplot(2, 1, 1)
    # sns.kdeplot(mcmc_output_0[:,1], fill = True)
    # plt.plot(np.linspace(10, 21, 1000), triang.pdf(np.linspace(10, 21, 1000), c = 4/11, loc = 10, scale = 11), color = 'k')
    # plt.axvline(x=14, color = 'r')
    # #plt.legend(['Posterior Distribution', 'Prior Distribution'], fontsize = 14)
    # # plt.xlabel('')
    # plt.ylabel('Density', fontsize=16)
    # plt.xticks(fontsize = 16)
    # plt.yticks([])
    # plt.xlabel('Time to Recovery (days)', fontsize = 20)
    
    # plt.subplot(2, 1, 2)
    # sns.kdeplot(mcmc_output_0[:,1], fill = True)
    # plt.plot(np.linspace(5, 50, 1000), triang.pdf(np.linspace(5, 50, 1000), c = 5/9, loc = 5, scale = 45), color = 'k')
    # plt.axvline(x=30, color = 'r')
    # # plt.xlabel('')
    # plt.ylabel('Density', fontsize=16)
    # plt.xticks(fontsize = 16)
    # plt.yticks([])
    # plt.xlabel('Percent Advancing to Symptoms', fontsize = 20)
    
    # # plt.subplot(3, 3, 4)
    # # sns.kdeplot(mcmc_output_params[:,3], fill = True)
    # # plt.plot(np.linspace(0.8, 2.5, 1000), triang.pdf(np.linspace(0.8, 2.5, 1000), c = 0.12, loc=0.8, scale=1.7), color = 'k')
    # # plt.axvline(x=1.004, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('$R_t$', fontsize = 20)
    
    # # plt.subplot(3, 3, 5)
    # # sns.kdeplot(mcmc_output_params[:,4], fill = True)
    # # plt.plot(np.linspace(0, 0.011, 1000), triang.pdf(np.linspace(0, 0.01, 1000), c = 0.05, loc=0, scale=0.01), color = 'k')
    # # plt.axvline(x=0.0005, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('Symptom Case Fatality Ratio', fontsize = 20)
    
    
    # # plt.subplot(3, 3, 6)
    # # sns.kdeplot(mcmc_output_params[:,5], fill = True)
    # # plt.plot(np.linspace(3, 12, 1000), triang.pdf(np.linspace(3, 12, 1000), c = 2/9, loc=3, scale=9), color = 'k')
    # # plt.axvline(x=5, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('Days to Incubation', fontsize = 20)
     
    # # plt.subplot(3, 3, 7)
    # # sns.kdeplot(mcmc_output_params[:,6], fill = True)
    # # plt.plot(np.linspace(0.7, 0.9, 1000), triang.pdf(np.linspace(0.7, 0.9, 1000), c = 0.5, loc=0.7, scale=0.2), color = 'k')
    # # plt.axvline(x=0.8, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('Test Sensitivity', fontsize = 20)
    
    # # plt.subplot(3, 3, 8)
    # # sns.kdeplot(mcmc_output_params[:,7], fill = True)
    # # plt.plot(np.linspace(0.95, 1, 1000), triang.pdf(np.linspace(0.95, 1, 1000), c =0.6,loc=0.95, scale=0.05), color = 'k')
    # # plt.axvline(x=0.98, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('Test Specificity', fontsize = 20)
    
    # # plt.subplot(3, 3, 9)
    # # sns.kdeplot(mcmc_output_params[:,8], fill = True)
    # # plt.plot(np.arange(0,100), nbinom.pmf(np.arange(0,100),5,0.25), 'k')
    # # plt.axvline(x=12, color = 'r')
    # # # plt.xlabel('')
    # # plt.ylabel('Density', fontsize=16)
    # # plt.xticks(fontsize = 16)
    # # plt.yticks([])
    # # plt.xlabel('New Infections per Exogenous Shock', fontsize = 20)
    
    # fig.tight_layout()
    # fig.align_labels()
    
    # plt.show()
    
    fig, axs = plt.subplots(3,3)
    fig.set_figheight(17)
    fig.set_figwidth(16)
    #labels=['Percent Advancing to Symptoms','Days to Incubation','Test Sensitivity','Test Specificity']

    labels=['Frequency of Exogenous Shocks (days)', 'Time to Recovery (days)','Percent Advancing to Symptoms', '$R_t$','Symptom Case Fatality Ratio','Days to Incubation','Test Sensitivity','Test Specificity','New Infections per Exogenous Shock']
    for i in range(Num_Params):
        plt.subplot(3, 3, i+1)
        plt.plot(mcmc_output_0[:,i])
        plt.plot(mcmc_output_1[:,i])
        plt.xlabel(labels[i])
    plt.show()
    print('Benchmark Test Completed')
    
    return()

def history_plots_and_convergence(mcmc_output_0, mcmc_output_1, convergence_threshold,  Num_Params):
    #This function plots the posteriors against the priors to ensure when the benchmark test is run they are the same
    
    mcmc_output_0[:,0] = np.array(mcmc_output_0[:,0], dtype = int)
    mcmc_output_1[:,0] = np.array(mcmc_output_1[:,0], dtype = int)
    
    # chains = [ChainConsumer() for i in range(len(mcmc_output_1[0,:]))]
    # converged = [False for i in range(len(mcmc_output_1[0,:]))]
    # end = len(mcmc_output_1[:,0])
    # for i in range(0, len(mcmc_output_1[0,:])):
        
    #     if i==0: 
    #         name = 'Freq_ES' 
        
    #     elif i==1:  
    #         name = "Time_Recov" 
    #     elif i==2: 
    #         name = 'Percent_advance' 
    #     elif i==3: 
    #         name = 'Rt' 
    #     elif i==4: 
    #         name = 'Symptom_fatal' 
    #     elif i==5: 
    #         name = 'Days_to_Incubation'  
    #     elif i==6: 
    #         name = 'Test_Sensitivity'
    #     elif i==7: 
    #         name = 'Test_Specificity' 
    #     elif i==8: 
    #         name ='New_Infect'
            
    #     chains[i].add_chain(np.concatenate((mcmc_output_0[:convergence_threshold,i], mcmc_output_1[:convergence_threshold,i])), walkers=2, name=name)
    #     converged = chains[i].diagnostic.gelman_rubin()
    
    fig, axs = plt.subplots(3,3)
    fig.set_figheight(17)
    fig.set_figwidth(16)
    #labels=['Percent Advancing to Symptoms','Days to Incubation','Test Sensitivity','Test Specificity']

    labels=['Frequency of Exogenous Shocks (days)', 'Time to Recovery (days)','Percent Advancing to Symptoms', '$R_t$','Symptom Case Fatality Ratio','Days to Incubation','Test Sensitivity','Test Specificity','New Infections per Exogenous Shock']
    for i in range(Num_Params):
        plt.subplot(3, 3, i+1)
        plt.plot(mcmc_output_0[:,i])
        plt.plot(mcmc_output_1[:,i])
        plt.xlabel(labels[i])
    plt.show()
    
    return()

def log_likelihood_weights(RIT_testing_data, Use_weights):
    
    if Use_weights == False: # if we do not need weights we just initailize the weight vector to one
        
        weight_norm = np.ones(len(RIT_testing_data.iloc[:,2])) # initializes a vector of ones that is the same length as the testing vector
   
    if Use_weights == True: # if we do want to use weights we will calculate them based on the number of tests we administer each day
        
        weights = np.zeros(len(RIT_testing_data.iloc[:,2])) # intialize a vector of zeros to fill
        
        # Initial set
        for i in range(len(RIT_testing_data.iloc[:,2])): # cycle through each day of testing to find the weights
            if RIT_testing_data.iloc[i,2]== 0:
                weights[i] = 0
            else:
                weights[i] = 1 / (RIT_testing_data.iloc[i,2] * RIT_testing_data.iloc[i,2]) # we set the weights to initially be 1\number of tests administered that day
        # Normalization of Weights
        weight_sum = np.sum(weights) # find sum of weights
        weight_norm = weights/weight_sum # divide each weight by sum of weights so all add to one
    
    return(weight_norm)