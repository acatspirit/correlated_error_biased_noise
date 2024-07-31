import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg
import CompassCodes as cc
import csv
import pandas as pd
import os



def depolarizing_err(p, H, eta=0.5):
    """Generates the error vector for one shot according to depolarizing noise model.
       Args:
       - p: Error probability.
       - num_qubits: Number of qubits.
       - eta: depolarizing channel bias. Recover unbiased depolarizing noise eta = 0.5. 
                Px, py, pz are determined according to 2D Compass Codes paper (2019) defn of eta
       
       Returns:
       - A list containing error vectors for no error, X, Z, and Y errors.
    """
    num_qubits = H.shape[1]
    # Error vectors for I, X, Z, and Y errors
    errors = np.zeros((2, num_qubits), dtype=int)

    # p = px + py + pz, px=py, eta = pz/(px + py)
    px = 0.5*p/(1+eta)
    pz = p*(eta/(1+eta))
    probs = [1 - p, px, pz, px]  # Probabilities for I, X, Z, and Y errors

    # Randomly choose error types for all qubits
    # np.random.seed(10)
    choices = np.random.choice(4, size=num_qubits, p=probs)
    # Assign errors based on the chosen types
    errors[0] = np.where((choices == 1) | (choices == 3), 1, 0)  # X or Y error
    errors[1] = np.where((choices == 2) | (choices == 3), 1, 0)  # Z or Y error
    return errors


def decoding_failures(H, L, p, eta, shots, err_type):
    """ finds the number of logical errors after decoding
        H - parity check matrix
        L - logical operator vector
        p - probability of error
        eta - depolarizing channel bias
        shots - number of shots
        err_type - the type of error that you hope to decode, X = 0, Z = 1
    """
    M = Matching.from_check_matrix(H)
    # get the depolarizing error vector 
    
    err_vec = [depolarizing_err(p, H, eta=eta)[err_type] for _ in range(shots)]
    # generate the syndrome for each shot
    syndrome_shots = err_vec@H.T%2
    # the correction to the errors
    correction = M.decode_batch(syndrome_shots)
    num_errors = np.sum((correction+err_vec)@L%2)
    return num_errors


def decoding_failures_correlated(H_x, H_z, L_x, L_z, p, eta, shots):
    """ Finds the number of logical errors after decoding.
        H_x - X parity check matrix for Z errors
        H_z - Z parity check matrix for X errors
        L_x - logical operator vector for X operators
        L_z - logical operator vector for X operators
        p - probability of error
        eta - depolarizing channel bias.
        shots - number of shots
    """
    # create a matching graph
    M_z = Matching.from_check_matrix(H_z)
    
    # Generate error vectors
    err_vec = [depolarizing_err(p, H_x, eta=eta) for _ in range(shots)]
    err_vec_x = np.array([err[0] for err in err_vec])
    err_vec_z = np.array([err[1] for err in err_vec])
    
    # Syndrome for Z errors and decoding
    syndrome_z = err_vec_x @ H_z.T % 2
    correction_z = M_z.decode_batch(syndrome_z)
    num_errors_x = np.sum((correction_z + err_vec_x) @ L_z % 2)
    
    # Prepare weights and syndrome for X errors
    updated_weights = np.logical_not(correction_z).astype(int)
    syndrome_x = err_vec_z @ H_x.T % 2
    
    # Decode X errors 
    num_errors_z = 0
    for i in range(shots):
        M_x = Matching.from_check_matrix(H_x, weights=updated_weights[i])
        correction_x = M_x.decode(syndrome_x[i])
        num_errors_z += np.sum((correction_x + err_vec_z[i]) @ L_x % 2)

    return num_errors_x, num_errors_z

def decoding_failures_total(H_x, H_z, L_x, L_z, p, eta, shots):
    """ Finds the number of logical errors after decoding.
        H_x - X parity check matrix for Z errors
        H_z - Z parity check matrix for X errors
        L_x - logical operator vector for X operators
        L_z - logical operator vector for X operators
        p - probability of error
        eta - depolarizing channel bias
        shots - number of shots
    """
    # create a matching graph
    M_z = Matching.from_check_matrix(H_z)
    M_x = Matching.from_check_matrix(H_x)
    
    # Generate error vectors
    err_vec = [depolarizing_err(p, H_x, eta=eta) for _ in range(shots)]
    err_vec_x = np.array([err[0] for err in err_vec])
    err_vec_z = np.array([err[1] for err in err_vec])
    
    # Syndrome for Z errors and decoding
    syndrome_z = err_vec_x @ H_z.T % 2
    correction_z = M_z.decode_batch(syndrome_z)
    num_errors_x = np.sum((correction_z + err_vec_x) @ L_z % 2)
    
    # Syndrome for X errors and decoding
    syndrome_x = err_vec_z @ H_x.T % 2
    correction_x = M_x.decode_batch(syndrome_x)
    num_errors_z = np.sum((correction_x + err_vec_z) @ L_x % 2)
    
    return num_errors_x, num_errors_z


#l = 3,4,6
# d = 3,5,7
# p_list = np.linspace(0.01, 0.5, 20)
# d_list = [3,5,7]
# l = 3
# num_shots = 10000

# log_err_list_x = []
# log_err_list_z = []
# for d in d_list:
#     print(f"simulating d={d}")

#     compass_code = cc.CompassCode(d=d, l=l)

#     H_x, H_z = compass_code.H['X'], compass_code.H['Z']
#     log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']
    
#     log_errors_x = []
#     log_errors_z = []
#     for p in p_list:
#         num_errors_x = decoding_failures(H_z, log_z, p, num_shots, 0)
#         log_errors_x.append(num_errors_x/num_shots)
        
#         num_errors_z = decoding_failures(H_x, log_x, p, num_shots, 1)
#         log_errors_z.append(num_errors_z/num_shots)

#     log_err_list_x.append(np.array(log_errors_x))
#     log_err_list_z.append(np.array(log_errors_z))

# # Create a figure with two subplots
# fig, (ax1,ax2 )= plt.subplots(1, 2, figsize=(12, 5))


# # Plot on the first subplot (ax1)
# for d, logical_errors_x in zip(d_list, log_err_list_x):
#     ax1.plot(2*p_list/3, logical_errors_x, label="d={}".format(d))
# ax1.set_title('X Errors')
# ax1.set_xlabel("Physical Error Rate")
# ax1.set_ylabel('Logical Error Rate')
# ax1.legend()
# ax1.grid(True)

# # Plot on the first subplot (ax2)
# for d, logical_errors_z in zip(d_list, log_err_list_z):
#     ax2.plot(2*p_list/3, logical_errors_z, label="d={}".format(d))
# ax2.set_title('Z Errors')
# ax2.set_xlabel("Physical Error Rate")
# ax2.set_ylabel('Logical Error Rate')
# ax2.legend()
# ax2.grid(True)

# plt.show()
#
# for generating a threshold graph for Z/X too 
#

num_shots = 1000000
d_list = [3,5,7,9,11]
l=6
p_list = np.linspace(0.01, 0.5, 20)
eta = 5.89
prob_scale = [2*0.5/(1+eta), (1+2*eta)/(2*(1+eta))] # the rate by which we double count errors for each type, X and then Z
log_err_list_x = []
log_err_list_z = []
log_err_indep_list_z = []
log_total_err_list = []

for d in d_list:
    print(f"simulating d={d}")
    compass_code = cc.CompassCode(d=d, l=l)
    H_x, H_z = compass_code.H['X'], compass_code.H['Z']
    log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']

    log_errors_x = []
    log_errors_z = []
    log_errors_indep_z = []
    log_total_err = []
    for p in p_list:
        num_errors_x,num_errors_z = decoding_failures_correlated(H_x, H_z, log_x, log_z, p, eta, num_shots)
        num_indep_x, num_indep_z = decoding_failures_total(H_x, H_z, log_x, log_z, p, eta, num_shots)
        log_errors_x.append(num_errors_x/num_shots)
        log_errors_z.append(num_errors_z/num_shots)
        log_errors_indep_z.append(num_indep_z/num_shots)
        log_total_err.append((num_indep_x+num_indep_z)/num_shots)
    
    log_err_list_x.append(np.array(log_errors_x))
    log_err_list_z.append(np.array(log_errors_z))
    log_err_indep_list_z.append(np.array(log_errors_indep_z))
    log_total_err_list.append(np.array(log_total_err))


data = [log_err_list_x, log_err_list_z, log_err_indep_list_z, log_total_err_list]
ind_dict = {1:'x', 2:'z', 3:'corr_z', 4:'total'}
folder = f"l{l}_shots{num_shots}"

if not os.path.exists(folder):
    os.makedirs(folder)

for ind, sublist in enumerate(data):
    df = pd.DataFrame(sublist)
    file_name = os.path.join(folder,f"{ind_dict[ind+1]}.csv")
    df.to_csv(file_name, index=False, header=False)




#################################################################
# l=2 # eta=0.50 # pzx=0.164 # pthr=0.143 # pz=0.095 # px=0.095 #
# l=3 # eta=1.67 # pzx=0.163 # pthr=0.174 # pz=0.142 # px=0.065 #
# l=4 # eta=3.00 # pzx=0.181 # pthr=0.199 # pz=0.174 # px=0.049 #
# l=5 # eta=4.26 # pzx=0.203 # pthr=0.217 # pz=0.195 # px=0.041 #
# l=6 # eta=5.89 # pzx=0.222 # pthr=0.233 # pz=0.213 # px=0.034 #
#################################################################

