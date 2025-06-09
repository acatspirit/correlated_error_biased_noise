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
        num_errors_x = np.sum((correction_x + err_vec_x) @ self.L_z % 2)
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_z @ self.H_x.T % 2
        correction_z = M_x.decode_batch(syndrome_x)
        num_errors_z = np.sum((correction_z + err_vec_z) @ self.L_x % 2)

        
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
                num_errors_xz_corr += np.sum((correction_xz_corr + err_vec_z[i]) @ self.L_x % 2)
            
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
                num_errors_zx_corr += np.sum((correction_zx_corr + err_vec_x[i]) @ self.L_z % 2)
            
            num_errors_corr = num_errors_zx_corr + num_errors_z
        
        num_errors_tot = num_errors_x + num_errors_z

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
        dem = circuit.detector_error_model(approximate_disjoint_errors=True) # what does the decompose do?
        matchgraph = Matching.from_detector_error_model(dem)
        sampler = circuit.compile_detector_sampler()
        syndrome, obersvable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matchgraph.decode_batch(syndrome)
        num_errors = np.sum(np.any(np.array(obersvable_flips) != np.array(predictions), axis=1))
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
            log_error_L.append(log_errors/num_shots)

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
            log_errors_z = decoder.get_log_error_circuit_level(p_list, "X", num_shots) # get the Z logical errors from X memory experiment
            log_errors_x = decoder.get_log_error_circuit_level(p_list, "Z", num_shots) # get the X logical errors from Z memory experiment


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


def concat_csv(folder_path, output_file):
    """Combines all CSV files is in folder 'folder_path' and writes them to one common 
        'output_file'. The CSV files in folder_path are deleted.
        in: folder_path - the folder that stores all the csv files to be combined
            output_file - the file that the CSV files will be combined into
        out: no output. The folder_path files are deleted and the output_file has the files in folder_path added to it
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

    # change this for XZ
    # all_data.loc[all_data['error_type'] == 'corr_z', 'error_type'] = 'CORR_ZX'

    
    all_data.to_csv(output_file, index=False)
    
    for file in data_files:
        os.remove(file)

def full_error_plot(full_df, curr_eta, curr_l, curr_num_shots, corr_type, file, loglog=False, averaging=True, circuit_level=False, plot_by_l=False):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots) 
                    # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
                    ]

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

def threshold_fit(x, pth, nu, a, b, c):
    p,d = x
    X = (d**(1/nu))*(p-pth)
    return c + b*X + a*X**2


def get_threshold(full_df, pth0, p_range, l, eta, corr_type, num_shots):
    """ returns the threshold and confidence given a df 
        in: df - the dataframe containing all data, filtered for one error_type, l eta, and probability range
        out: p_thr - a float, the probability where intersection of different lattice distances occurred
    """
    df = full_df[(full_df['p'] < pth0 + p_range) & ( full_df['p'] > pth0 - p_range) & (full_df['l'] == l) & (full_df['eta'] == eta) & (full_df['error_type'] == corr_type) & (full_df['num_shots'] == num_shots)]
    
    # get the p_list and d_list from the dataframe
    p_list = df['p'].to_numpy().flatten()
    d_list = df['d'].to_numpy().flatten()
    error_list = df['num_log_errors'].to_numpy().flatten()

    # run the fitting function
    popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 1, 1, 1, 1])
    
    pth = popt[0] # the threshold probability
    pth_error = np.sqrt(pcov[0][0])

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
    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])


    num_shots = 100000 # number of shots to sample
    circuit_data = True # whether circuit level or code cap data is desired
    d_list = [7, 9, 11]
    d_dict = {}
    l=3 # elongation parameter of compass code
    p_list = np.linspace(0.001, 0.01, 20)
    eta = 0.5 # the degree of noise bias
    corr_type = "CORR_ZX"
    if circuit_data:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_circuit_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_circuit_data.csv'
    else:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/corr_err_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'

    # d = 3
    # type_mem = "X" # type of memory experiment, X or Z
    # decoder = CorrelatedDecoder(eta, d, l, corr_type)
    # # print(decoder.H_x, decoder.H_z)
    # circuit = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, 0.01], type_mem) # change list of ps dependent on model
    # # circuit.make_elongated_circuit_from_parity()
    # print(circuit.circuit)

    # decoder.get_log_error_circuit_level(p_list, type_mem, num_shots)

    
    # run this to get data from the dcc
    write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type, circuit_data=circuit_data)
    # run this once you have data and want to combo it to one csv
    # concat_csv(folder_path, output_file)


    # threshold today - 0.2075 ZX, 0.217
    # threshold old - 0.20 ZX, 0.22 


    # Load and filter the data
    # df = pd.read_csv(output_file)
    # df = df[(df['num_shots'] == num_shots) & (df['eta'] == eta)]

    # d_list = sorted(df['d'].unique())     # e.g., [7, 9, 11]

    # l_list = sorted(df['l'].unique())     # e.g., [2, 3, 4]

    # fig, axes = plt.subplots(1, len(d_list), figsize=(15, 4), sharex=True, sharey=True)

    # # Ensure axes is always iterable
    # if len(d_list) == 1:
    #     axes = [axes]

    # for col, d in enumerate(d_list):
    #     ax = axes[col]
    #     d_df = df[df['d'] == d]
    #     for l in l_list:
    #         l_df = d_df[d_df['l'] == l]
    #         l_df_averaged = shots_averaging(num_shots, l, eta, corr_type, l_df, output_file)
    #         l_df_averaged = l_df_averaged.sort_values(by='p')

    #         ax.plot(l_df_averaged['p'], l_df_averaged['num_log_errors'], label=rf"$n = {d},\ \ell = {l}$", marker='o')

    #     ax.set_title(f"$n = {d}$", fontsize=16)
    #     ax.set_xlabel(r"$p_i$", fontsize=14)
    #     if col == 0:
    #         ax.set_ylabel(r"$p_L$", fontsize=14)

    #     ax.grid(True)
    #     ax.legend(fontsize=9)

    # fig.suptitle(f'Logical Error Rates (X + Z memory errors) for $\\eta = {eta}$ and num_shots = {num_shots}', fontsize=18)
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

    # threshold_plot(df, p_th_init, p_diff, eta, l, num_shots, "Z", output_file, loglog=True, averaging=True,show_threshold=True)
    # full_error_plot(df, eta, l, num_shots, corr_type, output_file, loglog=False, averaging=True)



#################################################################
# l=2 # eta=0.50 # pzx=0.164 # pthr=0.143 # pz=0.095 # px=0.095 #
# l=3 # eta=1.67 # pzx=0.163 # pthr=0.174 # pz=0.142 # px=0.065 #
# l=4 # eta=3.00 # pzx=0.181 # pthr=0.199 # pz=0.174 # px=0.049 #
# l=5 # eta=4.26 # pzx=0.203 # pthr=0.217 # pz=0.195 # px=0.041 #
# l=6 # eta=5.89 # pzx=0.259 # pthr=0.221 # pz=0.216 # px=0.033 # from 1000000 shots
#################################################################

