import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from scipy import sparse, linalg
import CompassCodes as cc
import csv
import pandas as pd
import os
from datetime import datetime
import sys
import glob
from scipy.optimize import curve_fit
import clifford_deformed_cc_circuit as cc_circuit
import itertools
# from lmfit import Minimizer, Parameters, report_fit


##############################################
#
# CorrelatedDecoder class
#
##############################################

class CorrelatedDecoder:
    def __init__(self, eta, d, l, corr_type):
        self.eta = eta # the noise bias
        self.d = d # the distance of the compass code
        self.l = l # the elongation parameter
        self.corr_type = corr_type # the type of correlation for decoder

        compass_code = cc.CompassCode(d=self.d, l=self.l)
        self.H_x, self.H_z = compass_code.H['X'], compass_code.H['Z'] # parity check matrices from compass code class
        self.log_x, self.log_z = compass_code.logicals['X'], compass_code.logicals['Z'] # logical operators from compass code class

        



    def depolarizing_err(self, p):
        """Generates the error vector for one shot according to depolarizing noise model.
        Args:
        - p: Error probability.
        - num_qubits: Number of qubits.
        - eta: depolarizing channel bias. Recover unbiased depolarizing noise eta = 0.5. 
                    Px, py, pz are determined according to 2D Compass Codes paper (2019) defn of eta
        
        Returns:
        - A list containing error vectors for no error, X, Z, and Y errors.
        """
        H = self.H_x
        eta = self.eta

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
    
    #
    #
    # Decoder functions
    #
    #

    def decoding_failures(self,p, shots, error_type):
        """ finds the number of logical errors after decoding
            p - probability of error
            shots - number of shots
            error_type - the type of error that you hope to decode, X = 0, Z = 1
        """
        if error_type == "X": 
            H = self.H_x
            L = self.log_x
        elif error_type == "Z":
            H = self.H_z
            L = self.log_z
        M = Matching.from_check_matrix(H)
        # get the depolarizing error vector 
        err_vec = [self.depolarizing_err(p)[error_type] for _ in range(shots)]
        # generate the syndrome for each shot
        syndrome_shots = err_vec@H.T%2
        # the correction to the errors
        correction = M.decode_batch(syndrome_shots)
        num_errors = np.sum((correction+err_vec)@L%2)
        return num_errors


    def decoding_failures_correlated(self, p, shots):
        """ Finds the number of logical errors after decoding.
            p - probability of error
            shots - number of shots
            corr_type - CORR_XZ or CORR_ZX. Whether to return X then Z or Z then X correlated errors.
        """
        M_z = Matching.from_check_matrix(self.H_z)
        M_x = Matching.from_check_matrix(self.H_x)
        
        # Generate error vectors
        err_vec = [self.depolarizing_err(p) for _ in range(shots)]
        err_vec_x = np.array([err[0] for err in err_vec])
        err_vec_z = np.array([err[1] for err in err_vec])
        
        # Syndrome for Z errors and decoding
        syndrome_z = err_vec_x @ self.H_z.T % 2
        correction_x = M_z.decode_batch(syndrome_z)
        num_errors_x = np.sum((correction_x + err_vec_x) @ self.log_z % 2)
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_z @ self.H_x.T % 2
        correction_z = M_x.decode_batch(syndrome_x)
        num_errors_z = np.sum((correction_z + err_vec_z) @ self.log_x % 2)

        
        # Decode Z errors correlated
        if self.corr_type == "CORR_XZ": # correct Z errors after correcting X errors
            cond_prob = 0.5 # the conditional probability of Z error given a X error
            new_weight = np.log((1-cond_prob)/cond_prob)
            
            # Prepare weights and syndrome for X errors
            updated_weights = np.ones(correction_x.shape)
            updated_weights[np.nonzero(correction_x)] = new_weight
            
            num_errors_xz_corr = 0

            for i in range(shots):
                M_xz_corr = Matching.from_check_matrix(self.H_x, weights=updated_weights[i]) # updated weights set erasure to 0
                correction_xz_corr = M_xz_corr.decode(syndrome_x[i])
                num_errors_xz_corr += np.sum((correction_xz_corr + err_vec_z[i]) @ self.log_x % 2)
            
            num_errors_corr = num_errors_xz_corr + num_errors_x
        
        # Decode X errors correlated
        if self.corr_type == "CORR_ZX": # correct X errors after correcting Z errors
            cond_prob = 1/(2*self.eta+1) # the conditional probability of X error given a Z error
            new_weight = np.log((1-cond_prob)/cond_prob)

            # Prepare weights and syndrome for X errors
            updated_weights = np.ones(correction_z.shape)
            updated_weights[np.nonzero(correction_z)] = new_weight
            num_errors_zx_corr = 0

            for i in range(shots):
                M_zx_corr = Matching.from_check_matrix(self.H_z, weights=updated_weights[i]) # updated weights set erasure to 0
                correction_zx_corr = M_zx_corr.decode(syndrome_z[i])
                num_errors_zx_corr += np.sum((correction_zx_corr + err_vec_x[i]) @ self.log_z % 2)
            
            num_errors_corr = num_errors_zx_corr + num_errors_z
        
        num_errors_tot = num_errors_x + num_errors_z # do I need to change this?

        return num_errors_x, num_errors_z, num_errors_corr, num_errors_tot

    def decoding_failures_uncorr(self,p, shots):
        """ Finds the number of logical errors after decoding.
            p - probability of error
            shots - number of shots
        """
        # create a matching graph
        M_z = Matching.from_check_matrix(self.H_z)
        M_x = Matching.from_check_matrix(self.H_x)
        
        # Generate error vectors
        err_vec = [self.depolarizing_err(p) for _ in range(shots)]
        err_vec_x = np.array([err[0] for err in err_vec])
        err_vec_z = np.array([err[1] for err in err_vec])
        
        # Syndrome for Z errors and decoding
        syndrome_z = err_vec_x @ self.H_z.T % 2
        correction_z = M_z.decode_batch(syndrome_z)
        num_errors_x = np.sum((correction_z + err_vec_x) @ self.L_z % 2)
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_z @ self.H_x.T % 2
        correction_x = M_x.decode_batch(syndrome_x)
        num_errors_z = np.sum((correction_x + err_vec_z) @ self.L_x % 2)
        
        return num_errors_x, num_errors_z
    

    #
    #
    # Circuit functions
    #
    #

    def get_num_log_errors(self, circuit, num_shots):
        """
        Get the number of logical errors from a circuit phenom. model, not the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :return: number of logical errors
        """
        matching = Matching.from_stim_circuit(circuit)
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matching.decode_batch(detection_events)
        
        
        num_errors = 0
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors += 1
        return num_errors

    def get_num_log_errors_DEM(self, circuit, num_shots):
        """
        Get the number of logical errors from the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :return: number of logical errors
        """
        dem = circuit.detector_error_model() # what does the decompose do?
        matchgraph = Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syndrome, observable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matchgraph.decode_batch(syndrome)
        num_errors = np.sum(np.any(np.array(observable_flips) != np.array(predictions), axis=1))
        return num_errors

    def get_log_error_circuit_level(self, p_list, meas_type, num_shots):
        """
        Get the logical error rate for a list of physical error rates of gates at the circuit level
        :param p_list: list of p values
        :param meas_type: type of memory experiment(X or Z), stabilizers measured
        :param num_shots: number of shots to sample
        :return: list of logical error rates, opposite type of the measurement type (e.g. if meas_type is X, then Z logical errors are returned)
        """

        log_error_L = []
        for p in p_list:
            # make the circuit
            circuit = cc_circuit.CDCompassCodeCircuit(self.d, self.l, self.eta, [0.003, 0.001, p], meas_type) # change list of ps dependent on model

            log_errors = self.get_num_log_errors_DEM(circuit.circuit, num_shots)
            log_error_L.append(log_errors)

        return log_error_L

    def get_log_error_p(self, p_list, meas_type, num_shots):
        """ 
        Get the logical error rate for a list of physical error rates of gates at code cap using a circuit
        :param p_list: list of p values
        :param meas_type: type of memory experiment(X or Z), stabilizers measured
        :param num_shots: number of shots to sample
        :return: list of logical error rates
        """
        log_error_L = []
        for p in p_list:
            # make the circuit
            circuit = cc_circuit.CDCompassCodeCircuit(self.d, self.l, self.eta, [0.003, 0.001, p], meas_type)
            log_errors = self.get_num_log_errors(circuit.circuit, num_shots)
            log_error_L += [log_errors/num_shots]
        return log_error_L



############################################
#
# Functions for getting data from DCC
#
############################################

def get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data):
    """ Generate logical error rates for x,z, correlatex z, and total errors
        via MC sim in decoding_failures_correlated and add it to a shared pandas df
        
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
        
        out: a pandas df recording the logical error rate with all corresponding params

    """
    err_type = {0:"X", 1:"Z", 2:corr_type, 3:"TOTAL"}
    data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "num_log_errors":[], "time_stamp":[]}
    data = pd.DataFrame(data_dict)

    for d in d_list:
        if circuit_data:
            
                # circuit_x = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "X")
                # circuit_z = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "Z")
    
            decoder = CorrelatedDecoder(eta, d, l, corr_type)
            log_errors_z = decoder.get_log_error_circuit_level(p_list, "Z", num_shots) # get the Z logical errors from Z memory experiment
            log_errors_x = decoder.get_log_error_circuit_level(p_list, "X", num_shots) # get the X logical errors from X memory experiment


            for i,log_error in enumerate(log_errors_x):
                curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_Mem", "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
            for i,log_error in enumerate(log_errors_z):
                curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_Mem", "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)

        else:
            decoder = CorrelatedDecoder(eta, d, l, corr_type)

            for p in p_list:
                errors = decoder.decoding_failures_correlated(p, num_shots)
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



def write_data(num_shots, d_list, l, p_list, eta, ID, corr_type, circuit_data):
    """ Writes data from pandas df to a csv file, for use with SLURM arrays. Generates data for each slurm output on a CSV
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
            ID - SLURM input task_ID number, corresponds to which array element we run
    """
    data = get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data)
    if circuit_data:
        data_file = f'circuit_data/{ID}.csv'
        if not os.path.exists('circuit_data/'):
            os.mkdir('circuit_data')
    else:
        data_file = f'corr_err_data/{ID}.csv'
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


def concat_csv(folder_path, circuit_data):
    """Combines all CSV files is in folder 'folder_path' and writes them to one common 
        'output_file'. The CSV files in folder_path are deleted.
        in: folder_path - the folder that stores all the csv files to be combined
            output_file - the file that the CSV files will be combined into
        out: no output. The folder_path files are deleted and the output_file has the files in folder_path added to it
    """
    data_files = glob.glob(os.path.join(folder_path, '*.csv'))
    df_list_XZ = []
    df_list_ZX = []
    df_list_CL = []
    
    for file in data_files:
        df = pd.read_csv(file)
        if not circuit_data: # the error types are X, Z, CORR_XZ, CORR_ZX, TOTAL, want to classify based on CORR_XZ and CORR_ZX
            if 'CORR_XZ' in df['error_type'].values:
                df_list_XZ.append(df)
            elif 'CORR_ZX' in df['error_type'].values:
                df_list_ZX.append(df)
        else:
            df_list_CL.append(df) # the error types are X_Mem and Z_Mem
    
    if circuit_data:
        new_data_CL = pd.concat(df_list_CL, ignore_index=True)
    else:
        new_data_XZ = pd.concat(df_list_XZ, ignore_index=True)
        new_data_ZX = pd.concat(df_list_ZX, ignore_index=True)
    
    output_file_XZ = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
    output_file_ZX = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
    output_file_CL = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_circuit_data.csv'
    
    all_data_XZ = pd.DataFrame()
    all_data_ZX = pd.DataFrame()
    all_data_CL = pd.DataFrame()

    xz_exists = os.path.exists(output_file_XZ)
    zx_exists = os.path.exists(output_file_ZX)
    cl_exists = os.path.exists(output_file_CL)

    # Check if the output file already exists
    if xz_exists and not circuit_data:
        # If it exists, load the existing data
        existing_data = pd.read_csv(output_file_XZ)
        # Append the new data to the existing data
        all_data_XZ = pd.concat([existing_data, new_data_XZ], ignore_index=True)
    elif not xz_exists and not circuit_data:
        # If the file doesn't exist, the new data is the combined data
        all_data_XZ = new_data_XZ

    if zx_exists and not circuit_data:
        # If it exists, load the existing data
        existing_data = pd.read_csv(output_file_ZX)
        # Append the new data to the existing data
        all_data_ZX = pd.concat([existing_data, new_data_ZX], ignore_index=True)
    elif not circuit_data and not zx_exists:
        # If the file doesn't exist, the new data is the combined data
        all_data_ZX = new_data_ZX

    if cl_exists and circuit_data:
        # If it exists, load the existing data
        existing_data = pd.read_csv(output_file_CL)
        # Append the new data to the existing data
        all_data_CL = pd.concat([existing_data, new_data_CL], ignore_index=True)
    elif circuit_data and not cl_exists:
        # If the file doesn't exist, the new data is the combined data
        all_data_CL = output_file_CL

    
    all_data_XZ.to_csv(output_file_XZ, index=False)
    all_data_ZX.to_csv(output_file_ZX, index=False)
    all_data_CL.to_csv(output_file_CL, index=False)
    
    for file in data_files:
        os.remove(file)

def full_error_plot(full_df, curr_eta, curr_l, curr_num_shots, corr_type, file, loglog=False, averaging=True, circuit_level=False, plot_by_l=False):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    # filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)] 
                    # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
    filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta)]
   
    # Get unique error types and unique d values
    error_types = filtered_df['error_type'].unique()

    d_values = filtered_df['d'].unique()

    # Create a figure with subplots for each error type
    fig, axes = plt.subplots(len(error_types)//2, 2, figsize=(15, 5*len(error_types)//2))
    axes = axes.flatten()
    

    # Plot each error type in a separate subplot
    for i, error_type in enumerate(error_types):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
        ax.tick_params(axis='both', which='minor', labelsize=16) 
        error_type_df = filtered_df[filtered_df['error_type'] == error_type]
        # Plot each d value
        for d in d_values:
            d_df = error_type_df[error_type_df['d'] == d]
            if averaging:
                d_df_mean = shots_averaging(curr_num_shots, curr_l, curr_eta, error_type, d_df, file)
                if loglog:
                    ax.loglog(d_df_mean['p']*prob_scale[error_type], d_df_mean['num_log_errors'],  label=f'd={d}')
                    error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors']))
                    ax.fill_between(d_df_mean['p']*prob_scale[error_type], d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2)
                else:
                    ax.plot(d_df_mean['p']*prob_scale[error_type], d_df_mean['num_log_errors'],  label=f'd={d}')
            else:
                ax.scatter(d_df['p']*prob_scale[error_type], d_df['num_log_errors'], s=2, label=f'd={d}')

        
        ax.set_title(f'Error Type: {error_type}', fontsize=20)
        ax.set_xlabel('p', fontsize=14)
        ax.set_ylabel('num_log_errors', fontsize=20)
        ax.legend()

    fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()

def threshold_plot(full_df, p_th0, p_range, curr_eta, curr_l, curr_num_shots, corr_type, file, loglog=False, averaging=True, show_threshold=True):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['p'] > p_th0 - p_range)&(full_df['p'] < p_th0 + p_range)&(full_df['error_type'] == corr_type)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)]
    

    d_values = filtered_df['d'].unique()
    num_lines = len(d_values)
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, num_lines)
    colors = [cmap(val) for val in color_values]

    # Create a figure with subplots for each error type
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot each d value
    for i,d in enumerate(d_values):
        d_df = full_df[(full_df['d'] == d)&(full_df['error_type'] == corr_type)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)]
        if averaging:
            d_df_mean = shots_averaging(curr_num_shots, curr_l, curr_eta, corr_type, d_df, file)
            if loglog:
                ax.loglog(d_df_mean['p']*prob_scale[corr_type], d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
                error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors']))
                ax.fill_between(d_df_mean['p']*prob_scale[corr_type], d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2, color=colors[i])
            else:
                ax.plot(d_df_mean['p']*prob_scale[corr_type], d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
        else:
            ax.scatter(d_df['p']*prob_scale[corr_type], d_df['num_log_errors'], s=2, label=f'd={d}',color=colors[i])

    pth, pth_error = get_threshold(filtered_df, p_th0, p_range, curr_l, curr_eta, corr_type,curr_num_shots)
    
    if show_threshold:
        ax.vlines(pth, ymin=0, ymax=0.5, color='red', linestyles='--', label=f'pth = {pth:.3f} +/- {pth_error:.3f}')
    
    ax.set_title(f'Error Type: {corr_type}', fontsize=20)
    ax.set_xlabel('p', fontsize=14)
    ax.set_ylabel('num_log_errors', fontsize=20)
    ax.legend()

    fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()


def eta_threshold_plot(eta_df):
    """Make a single figure with a 2-column grid of subplots.
    Each row corresponds to a different `l`, with CORR_XZ on left and CORR_ZX on right.
    """
    eta_values = sorted(eta_df['eta'].unique())
    l_values = sorted(eta_df['l'].unique())
    num_rows = len(l_values)

    # Set up colors
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, num_rows)
    l_colors = [cmap(val) for val in color_values]

    # Create figure and 2-column grid
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 2 * num_rows), sharex=True, sharey=True)

    for row_idx, l in enumerate(l_values):
        for col_idx, error_type in enumerate(['CORR_XZ', 'CORR_ZX']):
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
            mask = (eta_df['l'] == l) & (eta_df['error_type'] == error_type)
            df_filtered = eta_df[mask].sort_values(by='eta')

            eta_vals = df_filtered['eta'].to_numpy()
            pth_list = df_filtered['pth'].to_numpy()
            pth_error_list = df_filtered['stderr'].to_numpy()

            ax.errorbar(eta_vals, pth_list, yerr=pth_error_list,
                        label=f'l = {l}', color=l_colors[row_idx], marker='o', capsize=5)

            if row_idx == 0:
                ax.set_title(f"{error_type}", fontsize=16)

            if col_idx == 0:
                ax.set_ylabel(f"l = {l}\nThreshold $p_{{th}}$", fontsize=12)

            if row_idx == num_rows - 1:
                ax.set_xlabel("Noise Bias (η)", fontsize=12)

            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    plt.show()


# def threshold_fit(x, pth, nu, a, b, c):
#     p,d = x
#     X = (d**(1/nu))*(p-pth)
#     return c + b*X + a*X**2

def threshold_fit(x, pth, nu, c):
    p,d = x
    X = (d**(1/nu))*(p-pth)
    return c + X 


def get_threshold(full_df, pth0, p_range, l, eta, corr_type, num_shots):
    """ returns the threshold and confidence given a df 
        in: df - the dataframe containing all data, filtered for one error_type, l eta, and probability range
        out: p_thr - a float, the probability where intersection of different lattice distances occurred
    """
    print(f"Getting threshold for l = {l}, eta = {eta}, error type = {corr_type}, num_shots = {num_shots}")
    df = full_df[(full_df['p'] < pth0 + p_range) & ( full_df['p'] > pth0 - p_range) & (full_df['l'] == l) & (full_df['eta'] == eta) & (full_df['error_type'] == corr_type) & (full_df['num_shots'] == num_shots)]
    # df = full_df
    if df.empty:
        return 0, 0

    # get the p_list and d_list from the dataframe
    p_list = df['p'].to_numpy().flatten()
    d_list = df['d'].to_numpy().flatten()
    error_list = df['num_log_errors'].to_numpy().flatten()

    # run the fitting function
    # popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 0.5, 1, 1, 1])
    popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 1, 1])
    
    pth = popt[0] # the threshold probability
    pth_error = np.sqrt(np.trace(pcov))
    overfitting = np.linalg.cond(pcov)
    # print(f"Overfitting condition number: {overfitting}")
    # print(f"diag of covariance matrix: {np.diag(pcov)}")
    return pth, pth_error


def get_prob_scale(corr_type, eta):
    """ extract the amount to be scaled by given a noise bias and the type of error
    """
    prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), corr_type: 1, 'TOTAL':1, 'X_Mem':  1, 'Z_Mem': 1}
    return prob_scale




#
# for generating a threshold graph for Z/X too 
#

if __name__ == "__main__":
    # for simulation results
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) # will iter over the total slurm array size and points to where you are 

    print(f"Task ID: {task_id}")
    slurm_array_size = int(os.environ['SLURM_ARRAY_TASK_MAX']) # the size of the slurm array, used to determine how many tasks to run, currently 1000
    print(f"SLURM Array Size: {slurm_array_size}")
    l_eta_corr_type_arr = list(itertools.product([2,3,4,5,6],[1.5,2.5,3.5,4.5,6,7], ["CORR_XZ", "CORR_ZX"])) # list of tuples (l, eta, corr_type), currently 40
    reps = slurm_array_size//len(l_eta_corr_type_arr) # how many times to run file, num_shots each time
    p_th_init_dict = {(2,0.5, "CORR_ZX"):0.157, (2,1, "CORR_ZX"):0.149, (2,5, "CORR_ZX"):0.110,
                      (3,0.5, "CORR_ZX"):0.177, (3,1, "CORR_ZX"):0.178, (3,5, "CORR_ZX"):0.155,
                      (4,0.5, "CORR_ZX"):0.146, (4,1, "CORR_ZX"):0.173, (4,5, "CORR_ZX"):0.187,
                      (5,0.5, "CORR_ZX"):0.120, (5,1, "CORR_ZX"):0.148, (5,5, "CORR_ZX"):0.210,
                      (6,0.5, "CORR_ZX"):0.093, (6,1, "CORR_ZX"):0.109, (6,5, "CORR_ZX"):0.235,
                      (2,0.5, "CORR_XZ"):0.160, (2,1, "CORR_XZ"):0.167, (2,5, "CORR_XZ"):0.120,
                      (3,0.5, "CORR_XZ"):0.128, (3,1, "CORR_XZ"):0.165, (3,5, "CORR_XZ"):0.160,
                      (4,0.5, "CORR_XZ"):0.090, (4,1, "CORR_XZ"):0.145, (4,5, "CORR_XZ"):0.190,
                      (5,0.5, "CORR_XZ"):0.075, (5,1, "CORR_XZ"):0.110, (5,5, "CORR_XZ"):0.210,
                      (6,0.5, "CORR_XZ"):0.065, (6,1, "CORR_XZ"):0.090, (6,5, "CORR_XZ"):0.230,
                      (2,0.75,"CORR_XZ"): 0.149, (2,0.75,"CORR_ZX"):0.155, (2,2,"CORR_XZ"): 0.139,
                        (2,2,"CORR_ZX"): 0.122, (2,3,"CORR_XZ"): 0.127, (2,3,"CORR_ZX"): 0.115,
                        (2,4,"CORR_XZ"): 0.121, (2,4,"CORR_ZX"): 0.112, (3,0.75,"CORR_XZ"): 0.149,
                        (3,0.75,"CORR_ZX"): 0.176, (3,2,"CORR_XZ"): 0.177, (3,2,"CORR_ZX"): 0.175,
                        (3,3,"CORR_XZ"): 0.167, (3,3,"CORR_ZX"): 0.165, (3,4,"CORR_XZ"): 0.160,
                        (3,4,"CORR_ZX"): 0.160, (4,0.75,"CORR_XZ"): 0.114, (4,0.75,"CORR_ZX"): 0.159,
                        (4,2,"CORR_XZ"): 0.187, (4,2,"CORR_ZX"): 0.189, (4,3,"CORR_XZ"): 0.196,
                        (4,3,"CORR_ZX"): 0.196, (4,4,"CORR_XZ"): 0.192, (4,4,"CORR_ZX"): 0.192,
                        (5,0.75,"CORR_XZ"): 0.009, (5,0.75,"CORR_ZX"): 0.118, (5,2,"CORR_XZ"): 0.188,
                        (5,2,"CORR_ZX"): 0.189, (5,3,"CORR_XZ"): 0.206,(5,3,"CORR_ZX"): 0.205,
                        (5,4,"CORR_XZ"): 0.209,(5,4,"CORR_ZX"): 0.210,(6,0.75,"CORR_XZ"): 0.07,
                        (6,0.75,"CORR_ZX"): 0.092,(6,2,"CORR_XZ"): 0.185,(6,2,"CORR_ZX"): 0.180,
                        (6,3,"CORR_XZ"): 0.210,(6,3,"CORR_ZX"): 0.212,(6,4,"CORR_XZ"): 0.222,
                        (6,4,"CORR_ZX"): 0.222}

                      

    ind = task_id%len(l_eta_corr_type_arr) # get the index of the task_id in the l_eta__corr_type_arr

    l,eta, corr_type = l_eta_corr_type_arr[ind] # get the l and eta from the task_id

    print("l,eta,corr_type", l,eta, corr_type)
    print("reps", reps)
    print("ind", ind)

    num_shots = int(1e6//reps) # number of shots to sample
    # num_shots = 41666
    print("num_shots", num_shots)
    circuit_data = False # whether circuit level or code cap data is desired

    # for plotting
    # eta = 4
    # l = 6
    # corr_type = "CORR_XZ"
    # error_type = "CORR_XZ"

    # simulation
    d_list = [11,13,15,17,19]
    # p_th_init = p_th_init_dict[(l,eta,corr_type)]
    # p_th_init = 0.158
    # p_list = np.linspace(p_th_init-0.03, p_th_init + 0.03, 40)
    p_list = np.linspace(0.07, 0.3, 40)
    
    
    if circuit_data:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data/'
    #     if corr_type == "CORR_ZX":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_circuit_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_circuit_data.csv'
    else:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/corr_err_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'

    
    # run this to get data from the dcc
    write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type, circuit_data=circuit_data)
    # run this once you have data and want to combo it to one csv
    # concat_csv(folder_path, circuit_data)


    # threshold today - 0.2075 ZX, 0.217
    # threshold old - 0.20 ZX, 0.22 


    # Load and filter only X_mem and Z_mem
    # get all the thresholds and store the data in a csv



    # df = pd.read_csv(output_file)
    # df = pd.read_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv')
    # eta_threshold_plot(df)

    # threshold_d = {(2,0.75,"CORR_XZ"): 0.149, (2,0.75,"CORR_ZX"):0.155, (2,2,"CORR_XZ"): 0.139,
    #                     (2,2,"CORR_ZX"): 0.122, (2,3,"CORR_XZ"): 0.127, (2,3,"CORR_ZX"): 0.115,
    #                     (2,4,"CORR_XZ"): 0.121, (2,4,"CORR_ZX"): 0.112, (3,0.75,"CORR_XZ"): 0.149,
    #                     (3,0.75,"CORR_ZX"): 0.176, (3,2,"CORR_XZ"): 0.177, (3,2,"CORR_ZX"): 0.175,
    #                     (3,3,"CORR_XZ"): 0.167, (3,3,"CORR_ZX"): 0.165, (3,4,"CORR_XZ"): 0.160,
    #                     (3,4,"CORR_ZX"): 0.160, (4,0.75,"CORR_XZ"): 0.114, (4,0.75,"CORR_ZX"): 0.159,
    #                     (4,2,"CORR_XZ"): 0.187, (4,2,"CORR_ZX"): 0.189, (4,3,"CORR_XZ"): 0.196,
    #                     (4,3,"CORR_ZX"): 0.196, (4,4,"CORR_XZ"): 0.192, (4,4,"CORR_ZX"): 0.192,
    #                     (5,0.75,"CORR_XZ"): 0.009, (5,0.75,"CORR_ZX"): 0.118, (5,2,"CORR_XZ"): 0.188,
    #                     (5,2,"CORR_ZX"): 0.189, (5,3,"CORR_XZ"): 0.206,(5,3,"CORR_ZX"): 0.205,
    #                     (5,4,"CORR_XZ"): 0.209,(5,4,"CORR_ZX"): 0.210,(6,0.75,"CORR_XZ"): 0.07,
    #                     (6,0.75,"CORR_ZX"): 0.092,(6,2,"CORR_XZ"): 0.185,(6,2,"CORR_ZX"): 0.180,
    #                     (6,3,"CORR_XZ"): 0.210,(6,3,"CORR_ZX"): 0.212,(6,4,"CORR_XZ"): 0.222,
    #                     (6,4,"CORR_ZX"): 0.222}

    # for key in threshold_d.keys():
    #     l, eta, corr_type = key
    #     print("l,eta,corr_type", l,eta, corr_type)

    #     if corr_type == "CORR_ZX":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
    #     df = pd.read_csv(output_file)
    #     # threshold_d = {}

    #     p_th_init = p_th_init_dict[key]
    #     threshold = get_threshold(df, p_th_init, 0.03, l, eta, corr_type, num_shots)
    #     threshold_d[key] = threshold
    
    # threshold_df = pd.DataFrame(threshold_d)

    # threshold_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds.csv', index=False)

    





    # df = df[(df['num_shots'] == num_shots) & (df['eta'] == eta)]

    # threshold_plot(df, p_th_init, 0.03, eta, l, num_shots, error_type, output_file, loglog=True, averaging=True, show_threshold=True)


    # # Group by p, d, l and sum the num_log_errors to create 'tot_mem'
    # df_tot = df.groupby(['p', 'd', 'l'], as_index=False)['num_log_errors'].sum()

    # d_list = sorted(df_tot['d'].unique())   # e.g., [7, 9, 11]
    # l_list = sorted(df_tot['l'].unique())   # e.g., [2, 3, 4]

    # fig, axes = plt.subplots(1, len(d_list), figsize=(15, 4), sharex=True, sharey=True)

    # # Ensure axes is always iterable
    # if len(d_list) == 1:
    #     axes = [axes]

    # for col, d in enumerate(d_list):
    #     ax = axes[col]
    #     d_df = df_tot[df_tot['d'] == d]
    #     for l in l_list:
    #         l_df = d_df[d_df['l'] == l].sort_values(by='p')

    #         # If you're still using a custom averaging function, apply it here:
    #         l_df_averaged = shots_averaging(num_shots, l, eta, corr_type, l_df, output_file)
    #         l_df_averaged = l_df_averaged.sort_values(by='p')
    #         # ax.plot(l_df_averaged['p'], l_df_averaged['num_log_errors'], ...)

    #         ax.plot(l_df_averaged['p'], l_df_averaged['num_log_errors'], label=rf"$n = {d},\ \ell = {l}$", marker='o')

    #     ax.set_title(f"$n = {d}$", fontsize=16)
    #     ax.set_xlabel(r"$p_i$", fontsize=14)
    #     if col == 0:
    #         ax.set_ylabel(r"$p_L$", fontsize=14)

    #     ax.grid(True)
    #     ax.legend(fontsize=9)

    # fig.suptitle(f'Logical Error Rates ($X_{{mem}} + Z_{{mem}}$) for $\\eta = {eta}$ and num_shots = {num_shots}', fontsize=18)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.90])
    # plt.show()



    # df_larger_p = df[df['p'] > 0.05]
    # # df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    # # today = datetime.now().date()
    # # df_today = df[df['time_stamp'].dt.date != today]

    # # print(df['time_stamp'].dtype)
    # p_th_init = 0.187
    # p_diff = 0.03

    # threshold, confidence = get_threshold(df, p_th_init, p_diff, l, eta, corr_type)
    # print(threshold, confidence)

    # threshold_plot(df, 0.123, 0.03, 0.75, 5, num_shots, "CORR_XZ", output_file, loglog=True, averaging=True,show_threshold=True)
    # full_error_plot(df, eta, l, num_shots, corr_type, output_file, loglog=False, averaging=True)



#################################################################
# l=2 # eta=0.50 # pzx=0.164 # pthr=0.143 # pz=0.095 # px=0.095 #
# l=3 # eta=1.67 # pzx=0.163 # pthr=0.174 # pz=0.142 # px=0.065 #
# l=4 # eta=3.00 # pzx=0.181 # pthr=0.199 # pz=0.174 # px=0.049 #
# l=5 # eta=4.26 # pzx=0.203 # pthr=0.217 # pz=0.195 # px=0.041 #
# l=6 # eta=5.89 # pzx=0.259 # pthr=0.221 # pz=0.216 # px=0.033 # from 1000000 shots
#################################################################
