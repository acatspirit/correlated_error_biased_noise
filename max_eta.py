import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import optimize
import CompassCodes as cc
import csv
from compass_code_correlated_error import decoding_failures_correlated, write_data, get_data
from functools import partial
from lmfit import Minimizer, Parameters, report_fit
import os
import pandas as pd

def threshold_fit(p,a, b, c, e, pth, mu, d):
    return a + b*(d**(1/mu))*(p-pth) + c*((d**(1/mu))*(p-pth))**2 + e * ((d**(1/mu))*(p-pth))**3

def combined_residual(params, p_list, d_list, error_list):
    """ Gets the residuals for a global fit of data with many different d's
        in: params - the input parameters to the fitting model
            p_list - list of physical error probabilities to scan
            d_list - list of lattice distances
            error_list - the list of lists of logical errors for each distance scanning over each probability.
                        Obtained from get_data function.
        returns: The residuals for global and individual fit variables, designated in the get_threshold function
    """
    residuals = []
    print(error_list)
    # now error list should start at the p-val close to the guess ... no longer ind of 0 
    for i in range(len(d_list)):
        curr_d = d_list[i]

        # Get the local parameters for the current dataset
        a = params[f'a{curr_d}']
        b = params[f'b{curr_d}']
        c = params[f'c{curr_d}']
        d = params[f'd{curr_d}']
        e = params[f'e{curr_d}']
        
        # Get the shared global parameter
        pth = params['pth']
        mu = params['mu']
        
        # Calculate the model and residual
        model_value = threshold_fit(p_list, a, b, c, e, pth, mu, d)
        # model_value = threshold_fit(p_list, a, b, c, pth, mu, d)
        residuals.append(error_list[i] - model_value)
    return np.concatenate(residuals)

# find the intersection point of the lines within a range
def get_threshold(error_count_df, d_list, pth_0, p_range, return_all = False):
    """ Using the list of logical error rates for different distances in errors_array,
        finds the intersection in p_list of the lines
        in: error_count_df - the dataframe containing all data, filtered for one error_type
            d_list - list of lattice distances
            p_list - list of probabilites scanned over to produce errors_array
            pth_0 - the current threshold probability guess for designated logical error
            return_all - indicates whether or not to return all fitting parameters or just the probability threshold
        out: p_thr - a float, the probability where intersection of different lattice distances occurred
            (or) result - the minimized object with all variables for each iteration
        used "Confinement-Higgs transition in a disordered gauge theory and the
            accuracy threshold for quantum memory" (2002) section 4.2 by Wang, Harrington, Preskill for fitting model
    """

    params = Parameters()
    for curr_d in d_list:
        
        # local params to vary for each d
        params.add(f"a{curr_d}", value = 0)
        params.add(f"b{curr_d}", value = 1)
        params.add(f"c{curr_d}", value = 1)
        params.add(f"e{curr_d}", value = 0)

        # local param to fix
        params.add(f"d{curr_d}", value=curr_d, vary=False) # lattice size, have been calling this d

        # global params
        params.add("pth", value=pth_0, vary=True)
        params['pth'].min = pth_0 - p_range
        params['pth'].max = pth_0 + p_range

        params.add("mu", value=1, vary=True)

    p_list_zoomed = error_count_df['p']
    error_list = error_count_df['num_log_errors']


    minimizer = Minimizer(combined_residual, params, fcn_args=(p_list_zoomed, d_list, error_list))
    result = minimizer.minimize()
    
    
    if return_all:
        return result
    else:
        return result.params['pth'].value
    

def get_max_thresh_from_eta(params, num_shots, l, p_list, d_list, err_type, th_range, p_th0_list, data_file="corr_err_data.csv", new_data=True):
    """ FUNCTION TO BE MAXIMIZED. Returns the threshold with the current data, only eta to be iterated
        Inputs:
            num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
            err_type - string, can get the threshold for x, z, corr_z, or total errors
    """
    err_dict = {'x':0, 'z':1, 'corr_z':2, 'total':3}
    p_th0 = p_th0_list[err_dict[err_type]]

    eta = params['eta']

    if new_data:
        full_data_df = get_data(num_shots, d_list, l, p_list, eta)
    else:
        full_data_df = pd.read_csv(data_file)
    
    error_data_df = full_data_df[(full_data_df['error_type'] == err_type)]
    errors_near_th_df = error_data_df[(error_data_df['p'] < p_th0 + th_range) & (error_data_df['p'] > p_th0 - th_range)]

    curr_th = get_threshold(errors_near_th_df, d_list, p_th0, th_range)
    
    return curr_th

def threshold_minimization(eta, num_shots, l, p_list, d_list, err_type, th_range, p_th0_list, new_data=True, data_file='corr_err_data.csv'):
    return -get_max_thresh_from_eta(eta, num_shots, l, p_list, d_list, err_type, th_range, p_th0_list, new_data=new_data, data_file=data_file)

# minimize the negative of the intersection point and return param eta
def get_opt_eta(in_num_shots, in_l, init_eta, in_p_list, in_d_list, in_err_type, in_th_range, in_p_th0_list, show_result = True, new_data=True, data_file='corr_err_data.csv'):
    """ Returns the eta that produces the maximum p_threshold for a given l and d, along with
        other relevant parameters. 
        in: num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
        out: array of the optimal eta and corresponding thresholds ptotal, pz/x, pz, and px
    """
    params = Parameters()

    # params to optimize
    params.add('eta', value=init_eta, vary=True)

    # perform the minimization
    minimizer = Minimizer(threshold_minimization, params=params, fcn_args=(in_num_shots, in_l, in_p_list, in_d_list, in_err_type, in_th_range, in_p_th0_list))
    result = minimizer.minimize()

    max_p_th = -threshold_minimization(result.params, in_num_shots, in_l, in_p_list, in_d_list, in_err_type, in_th_range, in_p_th0_list, new_data=new_data, data_file=data_file)
    optimal_eta = result.params['eta'].value

    if show_result:
        single_error_graph(in_d_list, in_p_list, optimal_eta, in_num_shots, in_l, in_err_type, in_th_range, max_p_th)

    return optimal_eta, max_p_th

def single_error_graph(d_list, p_list, eta, num_shots, l, err_type, th_range, p_th, new_data=False, data_file='corr_err_data.csv'):
    """ Make a plot for one type of error with full p_list, fitting the range around p_th specified by th_range.
        If new_data is False, single_error_graph plots existing data from the full csv file. Otherwise, it generates
        and plots new data.

        in - num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
            err_type - string, x, z, z_corr, or total
            th_range - the window to be graphed around p_th
            p_th - the probability threshold
            new_data - specifies whether to create new data or to write to the csv file
    """
    # get the relavent data
    if new_data:
        data_df = get_data(num_shots, d_list, l, p_list, eta)
    else:
        data_df = pd.read_csv(data_file)
    

    # filter the df to contain only rows of interest
    error_data_df = data_df[(data_df['error_type'] == err_type & data_df['num_shots'] == num_shots)]
    error_data_near_th_df = error_data_df[(error_data_df['p'] < p_th + th_range ) & ( error_data_df['p'] > p_th - th_range)]
    prob_scale = {'x': 2*0.5/(1+eta), 'z': (1+2*eta)/(2*(1+eta)), 'corr_z': 1, 'total':1}

    fig, ax = plt.subplots()
    ax.set_title(f"Compass Code Logical Error Rate, Eta={eta}")
    ax.set_xlabel("Physical Error Probability")
    ax.set_ylabel("Logical Error Rate")
    
    # Plot the data for all values of d
    for d in d_list:
        d_df = error_data_df[error_data_df['d'] == d]
        d_df_near_th = error_data_near_th_df[error_data_near_th_df['d'] == d]

        fit_result = get_threshold(d_df_near_th, d_list, p_th, th_range)
        curr_params= fit_result.params
        fit_values = threshold_fit(np.array(d_df_near_th['p']), curr_params[f'a{d}'].value, 
                                   curr_params[f'b{d}'].value, curr_params[f'c{d}'].value, 
                                   curr_params[f'e{d}'].value, p_th, curr_params['mu'].value, d)
        line, = ax.plot(d_df_near_th['p']*prob_scale[err_type], fit_values, linestyle='--', label=f'fit d={d}')
        color = line.get_color()
        ax.plot(d_df['p']*prob_scale[err_type], d_df['num_log_errors'], linestyle='dotted', color=color, label=f'd={d}')

    ax.legend()
    plt.show()



#
#
# Test zone
#
#


#
# Test the Minimizer
#

num_shots = 1000
l = 6
eta_0 = 5.89
p_list = np.linspace(0.2, 0.25, 50)
d_list = [7,9,11]
err_type = 'total'
err_dict = {'x':0, 'z':1, 'corr_z':2, 'total':3}
p_th_range = 0.01
p_th0_list = [0.033,0.216,0.259, 0.221]

csv_file = 'corr_err_data.csv'
df = pd.read_csv(csv_file)
p_range_df = df[(df['p'] > p_th0_list[err_dict[err_type]]-p_th_range) & df['p'] < p_th0_list[err_dict[err_type]]+p_th_range]

threshold = get_threshold(df, d_list, p_th0_list[0], p_th_range, return_all=False)
# opt_eta, max_p_th = get_opt_eta(num_shots, l, eta_0, p_list, d_list, err_type, p_th_range, p_th0_list, show_result=True)
# print(opt_eta, max_p_th)






# to work on
# - my correlated z function is markedly worse at guessing than the regular z function
# - accuracy of p_th guessing ... 
#   - when I set up the threshold so that it's higher than reality, it guesses higher ... otherwise lower?
#       - wider range ==> worse guess
#       - trying to set the guess to be higher and the range to be smaller ... see if it gets the right answer
#   - may have to do with the fact that these thresholds for z are not in range i would expect
# - test all the eta





#
# Testing the threshold finder
#
# num_shots = 5000
# d_list = [3,5,7]
# l=3
# p_list = np.linspace(0.01, 0.5, 500)
# eta =  1.67
# p_th0 = [0.065, 0.141, 0.2, 0.179] # x , z, z correlated, total
# full_data = get_data(num_shots, l, eta, p_list, d_list)
# p_range = 0.05
# get_var = {'x':0, 'z': 1, 'corr_z': 2, 'total':3}
# # result_list = [get_threshold(data[i], p_list, d_list, p_th0[i]) for i in range(len(data))]
# var = get_var['z']
# data = full_data[var]

# p_list_near_th = [p for p in p_list if p_th0[var] - p_range < p < p_th0[var] + p_range]
# data_near_th = [[data[d][i] for i in range(len(p_list)) if p_th0[var] - p_range < p_list[i] < p_th0[var] + p_range] for d in range(len(d_list))]
# result = get_threshold(data_near_th, p_list_near_th, d_list, p_th0[var],p_range,return_all=True)
# # result_x = get_threshold(data_near_th, p_list_near_th, d_list, p_th0[0],p_range,return_all=True)



# l = 6
# num_shots = 1000000
# eta = 5.89
# d_list = [3,5,7,9,11]
# p_list = np.linspace(0.01, 0.5, 20)
# prob_scale = [2*0.5/(1+eta), (1+2*eta)/(2*(1+eta))]
# ind_d = {1:'x', 2:'z', 3:'corr_z', 4:'total'}
# folder = f"l{l}_shots{num_shots}"
# files = os.listdir(folder)

# dfs = {}
# for file in files:
#     # Construct the full file path
#     file_path = os.path.join(folder, file)
    
#     # Read the CSV file into a DataFrame and append it to the list
#     df = pd.read_csv(file_path, header=None)
#     dfs[file[:-4]] = df

# data = dfs['z'].values


# print(p_list_near_th)
# result = get_threshold(data_near_th, p_list_near_th, d_list, p_th0[var], p_range, return_all=True)

# print(f"pth: {result.params[f'pth'].value} pm {result.params[f'pth'].stderr}") 
# print(f"mu: {result.params[f'mu'].value} pm {result.params[f'mu'].stderr}")


# for curr_d in d_list:
#     print(f"a{curr_d}: {result.params[f'a{curr_d}'].value} pm {result.params[f'a{curr_d}'].stderr}") 
#     print(f"b{curr_d}: {result.params[f'b{curr_d}'].value} pm {result.params[f'b{curr_d}'].stderr}") 
#     print(f"c{curr_d}: {result.params[f'c{curr_d}'].value} pm {result.params[f'c{curr_d}'].stderr}") 
#     print(f"d{curr_d}: {result.params[f'd{curr_d}'].value} pm {result.params[f'd{curr_d}'].stderr}") 
#     print(f"e{curr_d}: {result.params[f'e{curr_d}'].value} pm {result.params[f'e{curr_d}'].stderr}") 



# df

# plt.figure(figsize=(10, 5))
# for i, y in enumerate(data):
#     curr_d = d_list[i]

#     plt.plot(p_list_near_th, y, 'o', label=f'd={d_list[i]}')
#     plt.plot(p_list_near_th, threshold_fit(p_list_near_th, result.params[f'a{curr_d}'], result.params[f'b{curr_d}'], result.params[f'c{curr_d}'], result.params[f'e{curr_d}'], \
#                                    result.params[f'pth'], result.params[f'mu'],result.params[f'd{curr_d}']), label=f'Fit {i}')
#     # plt.plot(p_list, threshold_fit(p_list, result.params[f'a{curr_d}'], result.params[f'b{curr_d}'], result.params[f'c{curr_d}'], \
#     #                                result.params[f'pth'], result.params[f'mu'],result.params[f'd{curr_d}']), label=f'Fit {i}')
# plt.legend()
# plt.show()

# in sim

# plt.figure(figsize=(10, 5))
# for i, y in enumerate(data):
#     curr_d = d_list[i]
#     plt.plot(p_list, y, 'o', label=f'd={d_list[i]}')
#     plt.plot(p_list_near_th, threshold_fit(p_list_near_th, result.params[f'a{curr_d}'], result.params[f'b{curr_d}'], result.params[f'c{curr_d}'], result.params[f'e{curr_d}'], \
#                                    result.params[f'pth'], result.params[f'mu'],result.params[f'd{curr_d}']), label=f'Fit {i}')
#     # plt.plot(p_list_near_th, threshold_fit(p_list_near_th, result.params[f'a{curr_d}'], result.params[f'b{curr_d}'], result.params[f'c{curr_d}'], \
#     #                                result.params[f'pth'], result.params[f'mu'],result.params[f'd{curr_d}']), label=f'Fit {i}')    
# plt.legend()
# plt.show()
# fitting well (the error is not good), but it's not returning what I thought it should return - I think it's returning the overall min of the quadratic, not the crossing point"