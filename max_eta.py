import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg, optimize
import CompassCodes as cc
import csv
import compass_code_correlated_error 
from functools import partial

def quadratic(x,a,b,c):
    return a*(x-b)**2 + c

# find the intersection point of the lines within a range
def get_threshold(errors_array, p_list, pth_0):
    """ Using the list of logical error rates for different distances in errors_array,
        finds the intersection in p_list of the lines
        in: errors_array - array of lists of logical error rates at different probabilities
            p_list - list of probabilites scanned over to produce errors_array
            pth_0 - a list of the threshold guesses for each of the error types in errors_array
        out: p_thr - a float, the probability where intersection occurred
    """
    # do a quadratic fit to the area where the threshold is
    # can also try linear fit - do this one on the log log plot
    # maybe take the average of pairwise intersections


    # make pairwise sets of points from the errors_array
    # log_x_errs = errors_array[0]
    # log_z_errs = errors_array[1]
    # log_corr_z_errs = errors_array[2]
    # total_errs = errors_array[3]

    # guess threshold
    pth_averages = []
    pth_error_averages = []
    
    for i in range(len(errors_array)):
        pth_list = []
        pth_err_list = []
        for j in range(1,len(errors_array[i]), 2):
            net_err = [max(a,b) for a,b in errors_array[i][j-1], errors_array[i][j]] # fix this line
            popt, pcov = optimize.curve_fit(quadratic, p_list, net_err, p0=[1, pth_0[i], 1], bounds= ([0, pth_0[i]-0.2, 0],[np.inf, pth_0[i]+0.2, np.inf]), maxfev=5000)
            pth_list += [popt[1]]
            pth_err_list += [pcov[1]]

        pth_averages.append(sum(pth_list)/len(pth_list))
        pth_error_averages.append(sum(pth_err_list)/len(pth_err_list))

    return pth_averages


def get_data(num_shots, l, eta, p_list, d_list):
    """ For a given l and eta, for many distances produces the number of logical errors at 
        various probabilities. Returns the logical error rate for X errors, Z errors,
        Z errors after X errors have been corrected, and Z and X errors in total.
        in: num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
    """
    prob_scale = [2*0.5/(1+eta), (1+2*eta)/(2*(1+eta))] # the rate by which we double count errors for each type, X and then Z
    log_err_list_x = []
    log_err_corr_list_z = []
    log_err_indep_list_z = []
    log_total_err_list = []

    for d in d_list:
        print(f"simulating d={d}")
        compass_code = cc.CompassCode(d=d, l=l)
        H_x, H_z = compass_code.H['X'], compass_code.H['Z']
        log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']

        log_errors_x = []
        log_corr_z = []
        log_errors_indep_z = []
        log_total_err = []
        for p in p_list:
            num_errors_x,num_corr_z = compass_code_correlated_error.decoding_failures_correlated(H_x, H_z, log_x, log_z, p, eta, num_shots)
            num_indep_x, num_indep_z = compass_code_correlated_error.decoding_failures_total(H_x, H_z, log_x, log_z, p, eta, num_shots)
            log_errors_x.append(num_indep_x/num_shots)
            log_corr_z.append(num_corr_z/num_shots)
            log_errors_indep_z.append(num_indep_z/num_shots)
            log_total_err.append((num_indep_x+num_indep_z)/num_shots)
        
        log_err_list_x.append(np.array(log_errors_x))
        log_err_corr_list_z.append(np.array(log_corr_z))
        log_err_indep_list_z.append(np.array(log_errors_indep_z))
        log_total_err_list.append(np.array(log_total_err))

    data = [log_err_list_x, log_err_indep_list_z, log_err_corr_list_z, log_total_err]

    # scale for current eta
    data = [np.array(data[0])*prob_scale[0], np.array(data[1])*prob_scale[1], np.array(data[2]), np.array(data[3])]
    return data

def get_max_thresh_from_eta(num_shots, l, eta, p_list, d_list):
    """ FUNCTION TO BE MAXIMIZED. Returns the threshold with the current data, only eta to be iterated
        Inputs:
            num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
    """
    data = get_data(num_shots, l, eta, p_list, d_list)
    curr_thresh = get_threshold(data, p_list)
    return curr_thresh

def threshold_minimization(num_shots, l, eta, p_list, d_list):
    return -get_max_thresh_from_eta(num_shots, l, eta, p_list, d_list)

# minimize the negative of the intersection point and return param eta
def get_opt_eta(in_num_shots, in_l, init_eta, in_p_list, in_d_list):
    """ Returns the eta that produces the maximum p_threshold for a given l and d, along with
        other relevant parameters. 
        in: num_shots - the number of experimental iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
        out: array of the optimal eta and corresponding thresholds ptotal, pz/x, pz, and px
    """
    fixed_threshold = partial(threshold_minimization, num_shots=in_num_shots, l=in_l, p_list=in_p_list, d_list=in_d_list)
    best_threshold = optimize.minimize(lambda vars: fixed_threshold(*vars), x0 = [init_eta])
    return best_threshold.x

#
#
# Test zone
#
#


#
# Testing the threshold 
#
num_shots = 100
d_list = [3,5,7,9,11]
l=2
p_list = np.linspace(0.01, 0.5, 20)
eta =  0.5
p_th0 = [0.1, 0.1, 0.16, 0.14] # x , z, z correlated, total

data = get_data(num_shots, l, eta, p_list, d_list)
print(data)
pth_list = get_threshold(data, p_list, p_th0)

print(pth_list)