import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg
import CompassCodes as cc
import csv
import pandas as pd
import os
from datetime import datetime
import sys
import glob

# sys.argv[1] to get the parameter from slurm sh file
# use seed to map the job to change filename (you can call seed one of the params = which job you're running)

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


def decoding_failures_correlated(H_x, H_z, L_x, L_z, p, eta, shots, corr_type):
    """ Finds the number of logical errors after decoding.
        H_x - X parity check matrix for Z errors
        H_z - Z parity check matrix for X errors
        L_x - logical operator vector for X operators
        L_z - logical operator vector for X operators
        p - probability of error
        eta - depolarizing channel bias.
        shots - number of shots
        corr_type - X or Z. Whether to return Z or X correlated errors
    """
    M_z = Matching.from_check_matrix(H_z)
    M_x = Matching.from_check_matrix(H_x)
    
    # Generate error vectors
    err_vec = [depolarizing_err(p, H_x, eta=eta) for _ in range(shots)]
    err_vec_x = np.array([err[0] for err in err_vec])
    err_vec_z = np.array([err[1] for err in err_vec])
    
    # Syndrome for Z errors and decoding
    syndrome_z = err_vec_x @ H_z.T % 2
    correction_x = M_z.decode_batch(syndrome_z)
    num_errors_x = np.sum((correction_x + err_vec_x) @ L_z % 2)
    
    # Syndrome for X errors and decoding
    syndrome_x = err_vec_z @ H_x.T % 2
    correction_z = M_x.decode_batch(syndrome_x)
    num_errors_z = np.sum((correction_z + err_vec_z) @ L_x % 2)
    
    
    
    
    # Decode Z errors correlated
    if corr_type == "Z":
        # Prepare weights and syndrome for X errors
        updated_weights = np.logical_not(correction_x).astype(int)
        
        num_errors_z_corr = 0

        for i in range(shots):
            M_x_corr = Matching.from_check_matrix(H_x, weights=updated_weights[i])
            correction_z_corr = M_x_corr.decode(syndrome_x[i])
            num_errors_z_corr += np.sum((correction_z_corr + err_vec_z[i]) @ L_x % 2)
        
        num_errors_corr = num_errors_z_corr

    if corr_type == "X":
        # Prepare weights and syndrome for X errors
        updated_weights = np.logical_not(correction_z).astype(int)
        num_errors_x_corr = 0

        for i in range(shots):
            M_z_corr = Matching.from_check_matrix(H_z, weights=updated_weights[i])
            correction_x_corr = M_z_corr.decode(syndrome_z[i])
            num_errors_x_corr += np.sum((correction_x_corr + err_vec_x[i]) @ L_z % 2)
        
        num_errors_corr = num_errors_x_corr

    # how do we make this perform the same as above

    # change the edges in the existing M_x graph
    # edges = M_x.edges()
    # for i in range(shots):
    #     for edge, w in zip(edges, updated_weights[i]):
    #         M_x.add_edge(edge[0], edge[1], None, weight=w, merge_strategy="replace")
    #     correction_z = M_x.decode(syndrome_x[i])
    #     num_errors_z_corr += np.sum((correction_z + err_vec_z[i]) @ L_x % 2)

    
    num_errors_tot = num_errors_x + num_errors_z

    return num_errors_x, num_errors_z, num_errors_corr, num_errors_tot

def decoding_failures_uncorr(H_x, H_z, L_x, L_z, p, eta, shots):
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

def get_data(num_shots, d_list, l, p_list, eta, corr_type):
    """ Generate logical error rates for x,z, correlatex z, and total errors
        via MC sim in decoding_failures_correlated and add it to a shared pandas df
        
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
        
        out: a pandas df recording the logical error rate with all corresponding params

    """
    err_type = {0:"x", 1:"z", 2:"corr_z", 3:"total"}
    data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "num_log_errors":[], "time_stamp":[]}
    data = pd.DataFrame(data_dict)

    for d in d_list:
        print(f"Running d is {d}")
        compass_code = cc.CompassCode(d=d, l=l)
        H_x, H_z = compass_code.H['X'], compass_code.H['Z']
        log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']

        for p in p_list:
            errors = decoding_failures_correlated(H_x, H_z, log_x, log_z, p, eta, num_shots, corr_type)
            for i in range(len(errors)):
                curr_row = {"d":d, "num_shots":num_shots, "p":p, "l": l, "eta":eta, "error_type":err_type[i], "num_log_errors":errors[i]/num_shots, "time_stamp":datetime.now()}
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
    return data

def shots_averaging(num_shots, l, eta, err_type, in_df, file):
    """For the inputted number of shots, averages those shots over the array length run on computing cluster.  
        in: num_shots - int, the number of monte carlo shots in the original simulation
            arr_len -  int, the number of jobs / averaging interval desired
            l - int, elongation parameter
            eta - float, noise bias
            err_type - the type of error to average
            df - the dataframe of interest. If None, read from the CSV file
    """
    if in_df is None:
        in_data = pd.read_csv(file)
        data = in_data[(in_data['num_shots'] == num_shots) & (in_data['l'] == l) &(in_data['eta'] == eta) & (in_data['error_type'] == err_type)]
    else:
        data = in_df
    data_mean = data.groupby('p', as_index=False)['num_log_errors'].mean()
    return data_mean



def write_data(num_shots, d_list, l, p_list, eta, ID, corr_type):
    """ Writes data from pandas df to a csv file, for use with SLURM arrays. Generates data for each slurm output on a CSV
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
            ID - SLURM input task_ID number, corresponds to which array element we run
    """
    data = get_data(num_shots, d_list, l, p_list, eta, corr_type)
    data_file = f'corr_err_data/{ID}.csv'
    # data_file = f"corr_err_data.csv"
    if not os.path.exists('corr_err_data/'):
        os.mkdir('corr_err_data')

    # Check if the CSV file exists
    if os.path.isfile(data_file):
        # If it exists, load the existing data
        past_data = pd.read_csv(data_file)
        # Append the new data
        all_data = pd.concat([past_data, data], ignore_index=True)
    else:
        # If it doesn't exist, the new data will be the combined data
        all_data = data
    # Save the combined data to the CSV file
    all_data.to_csv(data_file, index=False)


def concat_csv(folder_path, output_file):
    """Combines all CSV files is in folder 'folder_path' and writes them to one common 
        'output_file'. The CSV files in folder_path are deleted.
        in: folder_path - the folder that stores all the csv files to be combined
            output_file - the file that the CSV files will be combined into
        out: no output. The folder_path files are deleted and the output_file is added to
    """
    data_files = glob.glob(os.path.join(folder_path, '*.csv'))

    df_list = []
    for file in data_files:
        df = pd.read_csv(file)
        df_list.append(df)
    
    new_data = pd.concat(df_list, ignore_index=True)

    # Check if the output file already exists
    if os.path.exists(output_file):
        # If it exists, load the existing data
        existing_data = pd.read_csv(output_file)
        # Append the new data to the existing data
        all_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        # If the file doesn't exist, the new data is the combined data
        all_data = new_data
    
    all_data.to_csv(output_file, index=False)
    
    for file in data_files:
        os.remove(file)

def full_error_plot(full_df, curr_eta, curr_l, curr_num_shots, file,  averaging=True ):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = {'x': 0.5/(1+curr_eta), 'z': (1+2*curr_eta)/(2*(1+curr_eta)), 'corr_z': 1, 'total':1}

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots) 
                    # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
                    ]

    # Get unique error types and unique d values
    error_types = filtered_df['error_type'].unique()
    d_values = filtered_df['d'].unique()

    # Create a figure with subplots for each error type
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot each error type in a separate subplot
    for i, error_type in enumerate(error_types):
        ax = axes[i]
        error_type_df = filtered_df[filtered_df['error_type'] == error_type]
        # Plot each d value
        for d in d_values:
            d_df = error_type_df[error_type_df['d'] == d]
            if averaging:
                d_df_mean = shots_averaging(curr_num_shots, curr_l, curr_eta, error_type, d_df, file)
                ax.plot(d_df_mean['p']*prob_scale[error_type], d_df_mean['num_log_errors'],  label=f'd={d}')
            else:
                ax.scatter(d_df['p']*prob_scale[error_type], d_df['num_log_errors'], s=2, label=f'd={d}')

        
        ax.set_title(f'Error Type: {error_type}')
        ax.set_xlabel('p')
        ax.set_ylabel('num_log_errors')
        ax.legend()

    fig.suptitle(f'Logical Error Rates for eta ={curr_eta} and l={curr_l}')
    plt.tight_layout()
    plt.show()

def make_error_plot():
    """ Make a threshold plot for the specified error type
    """ 
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    



def get_prob_scale(error_type, eta):
    """ extract the amount to be scaled by given a noise bias and the type of error
    """
    prob_scale = {'x': 0.5/(1+eta), 'z': (1+2*eta)/(2*(1+eta)), 'corr_z': 1, 'total':1}
    return prob_scale[error_type]
#
# for generating a threshold graph for Z/X too 
#

if __name__ == "__main__":
    # task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

    num_shots = 10000
    d_list = [11,13,15,17,19]
    l=5 # elongation parameter of compass code
    p_list = np.linspace(0.01, 0.5, 40)
    eta = 1 # the degree of noise bias
    corr_type = "X"
    folder_path = 'corr_err_data/'
    output_file = 'x_corr_err_data.csv'

    # run this to get data from the dcc
    # write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type)
    # run this once you have data and want to combo it to one csv
    # concat_csv(folder_path, output_file)


    # to plot the data
    df = pd.read_csv(output_file)
    full_error_plot(df, eta, l, num_shots, output_file, averaging=True)
    # 13.0,10000.0,0.0225641025641025,3.0,1.67,x
    # print(df[(df['d'] == 13) & (df['num_shots'] == 10000.0) &(df['l'] == 3.0) &(df['eta'] == 1.67) &(df['error_type'] == 'x') &(df['p'] == 0.0225641025641025)])


    




#################################################################
# l=2 # eta=0.50 # pzx=0.164 # pthr=0.143 # pz=0.095 # px=0.095 #
# l=3 # eta=1.67 # pzx=0.163 # pthr=0.174 # pz=0.142 # px=0.065 #
# l=4 # eta=3.00 # pzx=0.181 # pthr=0.199 # pz=0.174 # px=0.049 #
# l=5 # eta=4.26 # pzx=0.203 # pthr=0.217 # pz=0.195 # px=0.041 #
# l=6 # eta=5.89 # pzx=0.259 # pthr=0.221 # pz=0.216 # px=0.033 # from 1000000 shots
#################################################################

