import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg
import CompassCodes as cc
import stim 
import pandas as pd
from compass_code_correlated_error import depolarizing_err


def clifford_deform_parity_mats(H_x, H_z, d, l):
    """ Add clifford deformation to parity matrices for elongated compass code with elongation l and
        distance d. Clifford deformations are added according to Julie's model (i.e. each weight 4
        X stabilizer has 2 H applied to antidiagonal)
        inputs:
            H_x - (scipy sparse mat) the X parity check matrix
            H_z - (scipy sparse mat) the Z parity check matrix
            d - (int) code distance, must be odd
            l - (int) elongation parameter
        returns: (np array) H_x and H_z with clifford deformations
    """
    return H_x, H_z

def make_circuit_from_parity_mat(H_x, H_z):
    """ Given a parity check matrix pair, generates a STIM circuit and detectors to implement the outlined code.
        Inputs:
            H - (scipy sparse mat) the parity check matrix
            Type - str - type == "X" is the X parity check matrix and produces those stabilizers
                         type == "Z" ' ' 
        Returns: (Stim circuit object) the circuit corresponding to the checks of the specified code

        Note - the X and Z circuits can be seperated in code-capacity model. Circuit level model will
                require integrating the circuits to avoid hook errors. See fig 3. of this paper: 
                https://arxiv.org/pdf/1404.3747
        TODO: change this function for non-CSS codes - how do I do this generally
    """
    # initialize circuit
    circuit = stim.Circuit()

    curr_ancilla = 0
    num_ancillas = 1 # change this based on the number of ancillas needed, maybe different for non-CSS codes



    #create the X parity checks
    rows, cols, values = sparse.find(H_x)

    for row in range(H_x.shape[0]):
        
        # Get the slice of the data corresponding to the current row
        row_start = H_x.indptr[row]
        row_end = H_x.indptr[row + 1]

        # Extract the column indices and data values for this row
        curr_cols = cols[row_start:row_end] 
        
        # get the order of the qubits for the surface code
        mid_qubit = len(curr_cols)//2
        first_half = curr_cols[0:mid_qubit]
        second_half = curr_cols[mid_qubit:]
        if len(curr_cols) == 4:
            sorted_qubits = np.append(np.sort(first_half)[::-1], np.sort(second_half)[::-1])
        else:
            sorted_qubits = np.append(np.sort(second_half)[::-1], np.sort(first_half)[::-1])

        
        
        # initialize the circuit
        circuit.append("R", curr_ancilla)
        circuit.append("RX", [q + 1 for q in sorted_qubits])


        circuit.append("H", curr_ancilla)

        for qubit in sorted_qubits:
            circuit.append("CX", [curr_ancilla, qubit + num_ancillas])

        circuit.append("H", curr_ancilla)
        circuit.append("MR", curr_ancilla)
        circuit.append("DETECTOR", stim.target_rec(-1))

    # create the Z parity checks
    # TODO: fix the detectors for the Z checks
    # rows, cols, values = sparse.find(H_z)

    # for row in range(H_z.shape[0]):
        
    #     # Get the slice of the data corresponding to the current row
    #     row_start = H_z.indptr[row]
    #     row_end = H_z.indptr[row + 1]

    #     # Extract the column indices and data values for this row
    #     curr_cols = cols[row_start:row_end] 

    #     # get the order of the qubits for the surface code
    #     mid_qubit = len(curr_cols)//2
    #     first_half = curr_cols[0:mid_qubit]
    #     second_half = curr_cols[mid_qubit:]
    #     if len(curr_cols) == 4:
    #         sorted_qubits = np.append(np.sort(first_half)[::-1], np.sort(second_half)[::-1])
    #     else:
    #         sorted_qubits = np.append(np.sort(second_half)[::-1], np.sort(first_half)[::-1])

    #     circuit.append("R", curr_ancilla)

    #     for qubit in sorted_qubits:
    #         circuit.append("CX", [qubit + num_ancillas, curr_ancilla])
    
    #     circuit.append("MR", curr_ancilla)
    #     circuit.append("DETECTOR", stim.target_rec(-1))
            
    return circuit


d = 3
l = 2
eta = 0.5
err_type = 1 # pick the Z errors 
shots = 1000
p_list = np.linspace(0.01, 0.5, 40)
# df = pd.df

compass_code = cc.CompassCode(d=d, l=l)
H_x, H_z = compass_code.H['X'], compass_code.H['Z']
log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']


curr_circuit = make_circuit_from_parity_mat(H_x, H_z)
dem = curr_circuit.detector_error_model(decompose_errors=True)
print(repr(curr_circuit))

# print(curr_circuit.compile_detector_sampler().sample(shots=4))


# doing the H_x first, Z errors
# matching_from_p = Matching.from_check_matrix(H_x)
# matching_from_c = Matching.from_stim_circuit(curr_circuit)

# matching_from_p.draw()
# plt.show()
# num_errors_p = []
# num_errors_c = []

# for p in p_list:
#     err_vec = [depolarizing_err(p, H_x, eta=eta) for _ in range(shots)]
#     err_vec_x = np.array([err[0] for err in err_vec])
#     err_vec_z = np.array([err[1] for err in err_vec])

#     # generate the syndrome for each shot
#     syndrome_shots = err_vec_z@H_x.T%2

#     # the correction to the errors
#     correction_p = matching_from_p.decode_batch(syndrome_shots)
#     correction_c = matching_from_c.decode_batch(syndrome_shots)

#     num_errors_p += [np.sum((correction_p+err_vec_z)@log_x%2)]
#     num_errors_c += [np.sum((correction_c+err_vec_z)@log_x%2)]


# plt.plot(p_list,num_errors_p, label="parity H" )
# plt.plot(p_list, num_errors_c, label="circuit")

# plt.show()

