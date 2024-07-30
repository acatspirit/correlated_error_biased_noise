import numpy as np
from pymatching import Matching
import stim
import matplotlib.pyplot as plt
from scipy import sparse, linalg

# create parity check matrix, H

def get_matching_stim(d, rounds=1, circuit_err_vec = [0]):
    """ Retrieve the surface code parity check matrix via stim's built in circuit
        in: d (int) - the distance of the surface code 
        out: matching graph (matching object) - the matching graph corresponding to the surface code
    """
    circuit = stim.Circuit.generated("surface_code", distance=d, rounds=rounds) # add errs later
    matching = Matching.from_stim_circuit(circuit)

    return matching


def get_parity(d):
    """ Get the H_x and the H_z for a surface code of distance d
        in: d (int, odd) - the distance of the surface code
        out: H_x (array) - the parity X check matrix, H_z (array) - the parity Z check matrix
    """
    num_qubits = d**2
    num_plaq = int(0.5*(d**2 - 1))

    # Populate the H_x matrix
    H_x = np.zeros((num_plaq,num_qubits))
    H_z = np.zeros((num_plaq,num_qubits))
    


    #
    # The H_x Check
    #
    row_count = 0
    q_pos = 0
    for i in range(num_plaq):
        # if row starts with a whole plaquette
        if row_count%2 == 0:

            # add whole plaquettes
            if (q_pos + 1)%d != 0:
                H_x[i][q_pos], H_x[i][q_pos+1], H_x[i][q_pos+d], H_x[i][q_pos+d+1] = 1,1,1,1
                q_pos += 2
            
            # add the end plaquette
            else:
                H_x[i][q_pos], H_x[i][q_pos+d] = 1,1
                q_pos += 1
                row_count += 1
        # if row starts with a half plaquette
        else:
            # add whole plaquettes
            if q_pos%d != 0:
                H_x[i][q_pos], H_x[i][q_pos+1], H_x[i][q_pos+d], H_x[i][q_pos+d+1] = 1,1,1,1
                q_pos += 2

                # iterate row when we have reached the final plaquette
                if q_pos % d == 0:
                    row_count += 1
            
            # add first plaquette
            else:
                H_x[i][q_pos], H_x[i][q_pos+d] = 1,1
                q_pos += 1
    #
    # The H_z Check
    #

    row_count = 0 # which row our current plaquette is in
    q_pos = 0   # the current qubit 
    edge = True # is our qubit positioned on the top or bottom of the surface?

    for i in range(num_plaq):
        
        # when our row begins with an X plaquette
        if row_count % 2 == 0:
            
            if edge:
                
                # since our row begins with an X plaquette, we include a Z on top
                if (q_pos-d*row_count) % 2 == 0:
                    H_z[i][q_pos], H_z[i][q_pos+1] = 1,1
                
                # Z plaquettes filling between X plaquettes
                else:
                    H_z[i][q_pos], H_z[i][q_pos+1], H_z[i][q_pos+d], H_z[i][q_pos+1+d] = 1,1,1,1
                
                # our next plaquette always begins at the next qubit
                q_pos += 1


                # when we go to the next row, we begin the next plaquette a qubit later
                if (q_pos+1)%d == 0:
                    q_pos += 1
            
            else:

                # when not on an edge, always alternate X and Z plaquettes
                H_z[i][q_pos], H_z[i][q_pos+1], H_z[i][q_pos+d], H_z[i][q_pos+1+d] = 1,1,1,1
                q_pos += 2 
        
        # when our row begins with an Z plaquette       
        else:
            
            if edge:
                # include a Z on top of the first X plaquette, which is one qubit later in an odd row
                if (q_pos-d*row_count) % 2 == 1:
                    H_z[i][q_pos+d], H_z[i][q_pos+1+d] = 1,1
                
                # otherwise tile alternating X and Z plaquettes
                else:
                    H_z[i][q_pos], H_z[i][q_pos+1], H_z[i][q_pos+d], H_z[i][q_pos+1+d] = 1,1,1,1
                
                # every qubit on an edge has a Z plaquette associated
                q_pos += 1
            
            else:

                # when not on an edge, always alternate Z and X plaquettes
                H_z[i][q_pos], H_z[i][q_pos+1], H_z[i][q_pos+d], H_z[i][q_pos+1+d] = 1,1,1,1
                q_pos += 2

                # check that we haven't finished the row, in which case the next Z plaquette is 2 qubits later
                if (q_pos+1)%d == 0:
                    q_pos += 2
        
        
        # iterate our row, and edge variables when applicable
        if q_pos - (row_count+1)*d >= 0: # change this so that if a mod d is passed, we increment rowcount
            row_count += 1

        if q_pos - d >= 0 and (q_pos + 2*d + 1) < d**2:
            edge = False
        else:
            edge = True
                      
    return sparse.csc_matrix(H_x), sparse.csc_matrix(H_z)


def depolarizing_err(p, H):
    """Generates the error vector for one shot according to depolarizing noise model.
       Args:
       - p: Error probability.
       - num_qubits: Number of qubits.
       
       Returns:
       - A list containing error vectors for no error, X, Z, and Y errors.
    """
    num_qubits = H.shape[1]
    # Error vectors for I, X, Z, and Y errors
    errors = np.zeros((2, num_qubits), dtype=int)
    probs = [1 - p, p / 3, p / 3, p / 3]  # Probabilities for I, X, Z, and Y errors

    # Randomly choose error types for all qubits
    # np.random.seed(10)
    choices = np.random.choice(4, size=num_qubits, p=probs)
    # Assign errors based on the chosen types
    errors[0] = np.where((choices == 1) | (choices == 3), 1, 0)  # X or Y error
    errors[1] = np.where((choices == 2) | (choices == 3), 1, 0)  # Z or Y error
    return errors


def decoding_failures(H, L, p, shots, err_type):
    """ finds the number of logical errors after decoding
        H - parity check matrix
        L - logical operator vector
        p - probability of error
        shots - number of shots
        err_type - the type of error that you hope to decode, X = 0, Z = 1
    """
    M = Matching.from_check_matrix(H)
    # get the depolarizing error vector 
    
    err_vec = [depolarizing_err(p, H)[err_type] for _ in range(shots)]
    # generate the syndrome for each shot
    syndrome_shots = err_vec@H.T%2
    # the correction to the errors
    correction = M.decode_batch(syndrome_shots)
    num_errors = np.sum((correction+err_vec)@L%2)
    return num_errors


def decoding_failures_total(H_x, H_z, L_x, L_z, p, shots):
    """ Finds the number of logical errors after decoding.
        H_x - X parity check matrix for Z errors
        H_z - Z parity check matrix for X errors
        L_x - logical operator vector for X operators
        L_z - logical operator vector for X operators
        p - probability of error
        shots - number of shots
    """
    # create a matching graph
    M_z = Matching.from_check_matrix(H_z)
    
    # Generate error vectors
    err_vec = [depolarizing_err(p, H_x) for _ in range(shots)]
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

#
# for generating a threshold graph
#

num_shots = 10000
d_list = [3,5,7]
p_list = np.linspace(0.01, 0.5, 20)
log_err_list_x = []
log_err_list_z = []
log_err_indep_list_z = []

for d in d_list:
    print(f"simulating d={d}")
    H_x,H_z = get_parity(d)
    # log_x, log_z = np.concatenate((np.ones(d), np.zeros(H_x.shape[1]-d))), np.concatenate((np.ones(d), np.zeros(H_z.shape[1]-d)))
    log_x, log_z = np.ones(H_x.shape[1]), np.ones(H_z.shape[1])

    log_errors_x = []
    log_errors_z = []
    log_errors_indep_z = []
    for p in p_list:
        num_errors_x,num_errors_z = decoding_failures_total(H_x, H_z, log_x, log_z, p, num_shots)
        num_indep_z = decoding_failures(H_x, log_x, p, num_shots, 1)
        log_errors_x.append(num_errors_x/num_shots)
        log_errors_z.append(num_errors_z/num_shots)
        log_errors_indep_z.append(num_indep_z/num_shots)
    
    log_err_list_x.append(np.array(log_errors_x))
    log_err_list_z.append(np.array(log_errors_z))
    log_err_indep_list_z.append(np.array(log_errors_indep_z))



# Create a figure with two subplots
fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot on the first subplot (ax1)
for d, logical_errors_x in zip(d_list, log_err_list_x):
    ax1.plot(2*p_list/3, logical_errors_x, label="d={}".format(d))
ax1.set_title('X Errors')
ax1.set_xlabel("Physical Error Rate")
ax1.set_ylabel('Logical Error Rate')
ax1.legend()
ax1.grid(True)

# Plot on the first subplot (ax2)
for d, logical_errors_z in zip(d_list, log_err_list_z):
    ax2.plot(p_list, logical_errors_z, label="d={}".format(d))
ax2.set_title('Z/X Errors')
ax2.set_xlabel("Physical Error Rate")
ax2.set_ylabel('Logical Error Rate')
ax2.legend()
ax2.grid(True)


# Plot on the first subplot (ax2)
for d, logical_errors_indep_z in zip(d_list, log_err_indep_list_z):
    ax3.plot(2*p_list/3, logical_errors_indep_z, label="d={}".format(d))
ax3.set_title('Z Errors')
ax3.set_xlabel("Physical Error Rate")
ax3.set_ylabel('Logical Error Rate')
ax3.legend()
ax3.grid(True)
# Adjust layout to prevent overlap
plt.tight_layout()

# Display the figure with subplots
plt.show()

#
# Testing the XZ Algorithm 3
#
# d = 5
# p = 0.2
# num_shots = 100

# H_x,H_z = get_parity(d)
# log_x, log_z = np.ones(H_x.shape[1]), np.ones(H_z.shape[1])

# # do X decoding
# z_matching = Matching.from_check_matrix(H_z)
# # get the depolarizing error vector 
# err_vec_x = np.random.binomial(1,p, H_z.shape[1])
# # generate the syndrome for each shot
# syndrome_shots_z = err_vec_x@H_z.T%2
# # the correction to the errors
# correction_z = z_matching.decode(syndrome_shots_z)


# # check that the error was corrected if it was swept over
# updated_weights = [0 if correction_z[i] == err_vec_x[i] else 1 for i in range(d**2)]

# x_matching = Matching.from_check_matrix(H_x, weights=updated_weights)
# err_vec_z = np.random.binomial(1,p, H_x.shape[1])
# syndrome_shots_x = err_vec_z@H_x.T%2
# correction_x = z_matching.decode(syndrome_shots_x)

# num_errors = np.sum((correction_x+err_vec_z)@log_x%2)

# print(num_errors)
# adjust weights, do another round of matching
# print weights, figure out what the weights are for your x graph
# make the weights 0 where there was an error that you correct - check the syndrome and go to that qubit in the matching graph

# find out which X errors actually flipped, decode
# where there was an X error, set the weight to cost nothing to include

# change the distances so that it reflects the fact there was a Z error - delete edges that alreaduy have errors?


# print(sz)

# decode Z errors


###############################
#
# Debugging 
#


# p = 0.5
# d = 3
# H_x,H_z = get_parity(d)
# log_x, log_z = np.ones(d**2), np.ones(d**2)

# x_matching = Matching.from_check_matrix(H_x)
# z_matching = Matching.from_check_matrix(H_z)


# Check one matrix against specific errors


# e1 = [1] + (H_x.shape[1]-1)*[0]
# print(e1)

# # for x errors
# print(f"syndrome should be: [1,0,0,0]. Syndrome is: {H_z@e1%2}")
# print(f"logicals should be: 1. logicals are: {log_z@e1%2}. Decoding adds: {z_matching.decode(H_z@e1%2)}")
# print(f"The correction gives us: {(e1 + z_matching.decode(H_z@e1%2))%2@log_z%2}")

# # # for z errors


# e2 = [i%2 for i in range(H_x.shape[1])]
# print(e2)
# # for x errors
# print(f"syndrome should be: [1,0,0,1]. Syndrome is: {H_z@e2%2}")
# print(f"logicals should be: 0. Logicals are: {log_z@e2%2}. Decoding adds: {z_matching.decode(H_z@e2%2)}")
# print(f"The correction gives us: {(e2 + z_matching.decode(H_z@e2%2)%2)@log_z%2}")
# # for z errors


# e3 = [1,1] + (H_x.shape[1]-2)*[0]
# print(e3)
# # for x errors
# print(f"syndrome should be: [0,1,0,0]. Syndrome is: {H_z@e3%2}")
# print(f"logicals should be: 0. Logicals are: {log_z@e3%2}. Decoding adds: {z_matching.decode(H_z@e3%2)}")
# print(f"The correction gives us: {(e3 + z_matching.decode(H_z@e3%2)%2)@log_z%2}")
# # for z errors


# e4 = [1,1,0,1] + (H_x.shape[1]-4)*[0]
# print(e4)

# # for x errors
# print(f"syndrome should be: [0,1,1,0]. Syndrome is: {H_z@e4%2}")
# print(f"logicals should be: 1. Logicals are: {log_z@e4%2}. Decoding adds: {z_matching.decode(H_z@e4%2)}")
# print(f"The correction gives us: {(e4 + z_matching.decode(H_z@e4%2)%2)@log_z%2}")


# Check the probability function against specific errors


