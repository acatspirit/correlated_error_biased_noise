import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg
import CompassCodes as cc
import stim 
import pandas as pd
from compass_code_correlated_error import depolarizing_err
import sys

#
# TODO: This code is unfinished. Currently, the surface_code_circuit_scratch file contains the working code that generates a surface code circuit.
# Next:
# (1) modify to work on compass code
# (2) modify to work on XZZX / ZXXZ codes
# (3) modify to work on clifford deformed compass codes

class CDCompassCodeCircuit:
    def __init__(self, d, l, eta, type):
        self.d = d
        self.l = l
        self.eta = eta
        self.code = cc.CompassCode(d=d, l=l)
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z']
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z']
        self.type = type # str "X" or "Z", indicates the type of memory experiment / which stabilizer to measure

        self.qubit_order_d = self.check_order_d_elongated(type)

    #
    # Helper functions to make surface code circuits
    #

    def check_order_d(self):
        """ Change this for longer codes
            Right now (from STIM ex): 
            HX: 0 - TR, 1 - TL, 2 - BR, 3 - BL
            HZ: 0 - TR, 1 - BR, 2 - TL, 3 - BL
        """
        if self.type == "X": H = self.H_x
        elif self.type == "Z": H = self.H_z

        plaq_d = self.convert_sparse_to_d(H)
        order_d = {0:[], 1:[], 2:[], 3:[]}
        d = int(np.sqrt(H.shape[1]))

        for plaq in plaq_d:
            q_list = plaq_d[plaq] # the list of qubits in the plaquette
            if type == "X":
                if len(q_list) == 2 and max(q_list) <= H.shape[1]//2: # if the two qubit stabilizer is on the top bndry
                    order_d[2] += [(q_list[1], plaq)]
                    order_d[3] += [(q_list[0], plaq)]
                elif len(q_list) == 2 and max(q_list) >= H.shape[1]//2: # if the two qubit stabilizer is on bottom 
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[0], plaq)]
                else: # length 4 plaquette
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[0], plaq)]
                    order_d[2] += [(q_list[3], plaq)]
                    order_d[3] += [(q_list[2], plaq)]
            if type == "Z": 
                if len(plaq_d[plaq]) == 2 and (q_list[0]-q_list[1])%d == 0 and q_list[0]%d == 0: # if the two qubit stabilizer is on the left bndry
                    order_d[0] += [(q_list[0], plaq)]
                    order_d[1] += [(q_list[1], plaq)]
                elif len(plaq_d[plaq]) == 2 and (q_list[0]-q_list[1])%d == 0 and q_list[0]%d != 0: # if the two qubit stabilizer is on right bndry 
                    order_d[2] += [(q_list[0], plaq)]
                    order_d[3] += [(q_list[1], plaq)]
                else: # length 4 plaquette
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[3], plaq)]
                    order_d[2] += [(q_list[0], plaq)]
                    order_d[3] += [(q_list[2], plaq)]
                    
        return order_d

    def check_order_d_elongated(self):
        """ New order based on zigzag pattern in PRA (101)042312, 2020
        """
        if self.type == "X": H = self.H_x
        elif self.type == "Z": H = self.H_z

        # create the order dictionary to store the qubit ordering for each plaquette
        order_d = {}
        for row in range(H.shape[0]):
            order_d[row] = []

        # get the qubit ordering for each plaquette
        for row in range(H.shape[0]):
            start = H.indptr[row]
            end = H.indptr[row+1]
            qubits = sorted(H.indices[start:end]) # the qubits in the plaquette
            
            if type == "Z":
                for i in range(len(qubits)//2):
                    match_qubit_ind = np.where(qubits == (qubits[i] + d))[0][0]
                    order_d[row] += [(qubits[i], row)]
                    order_d[row] += [(qubits[match_qubit_ind], row)]

            if type == "X":
                for qubit in qubits:
                    order_d[row] += [(qubit, row)]
        return order_d

    def qubit_to_plaq_d(self):
        """ Given a parity check matrix, returns a dictionary of qubits to plaquettes
            Returns: (dict) qubit to plaquette mapping
        """
        if self.type == "X": H = self.H_x
        elif self.type == "Z": H = self.H_z

        rows, cols, values = sparse.find(H)
        d = {}
        for i in range(len(cols)):
            q = cols[i]
            plaq = rows[i]

            if q not in d:
                d[q] = [plaq]
            else:
                d[q] += [plaq]
        return d

    def convert_sparse_to_d(self):
        if self.type == "X": H = self.H_x
        elif self.type == "Z": H = self.H_z

        rows, cols, values = sparse.find(H)
        d = {}

        for i in range(len(rows)):
            plaq = rows[i]
            qubit = cols[i]

            if plaq not in d:
                d[plaq] = [cols[i]]
            else:
                d[plaq] += [cols[i]]
        sorted_d = dict(sorted(zip(d.keys(),d.values())))
        return sorted_d


    def clifford_deform_parity_mats(self):
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
        return self.H_x, self.H_z
    


    ############################################
    # 
    # Functions to make the circuits
    # 
    ############################################  

    def make_circuit_from_parity(self, p_err):
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
            TODO: add elongation
        """
        H_x, H_z = self.H_x, self.H_z
        # make the circuit
        circuit = stim.Circuit()

        # get the qubit ordering
        plaq_d_x = self.convert_sparse_to_d()
        plaq_d_z = self.convert_sparse_to_d()
        
        order_d_x = self.check_order_d()
        order_d_z = self.check_order_d()
        
        qubit_d_x = self.qubit_to_plaq_d()
        qubit_d_z = self.qubit_to_plaq_d()
        
        # general parameters
        num_ancillas = len(plaq_d_x) + len(plaq_d_z)
        num_qubits = len(qubit_d_x)
        d = self.d

        data_q_x_list = [num_ancillas + q for q in list(qubit_d_x.keys())]
        data_q_z_list = [num_ancillas + q for q in list(qubit_d_z.keys())]
        data_q_list = data_q_x_list

        # convention - X plaqs first, then Z plaqs starting with 0
        full_plaq_L = range(len(plaq_d_x) + len(plaq_d_z))
        
        # reset the ancillas
        circuit.append("R", full_plaq_L)
        circuit.append("H", plaq_d_x)

        # reset the qubits
        for q in range(len(qubit_d_x)):
            if type == "X":
                circuit.append("RX", q + num_ancillas)
            if type == "Z":
                circuit.append("R", q + num_ancillas)
    
        
        for order in order_d_x: # go thru the qubits in order of gates
            q_x_list = order_d_x[order] # (qubit, ancilla)
            q_z_list = order_d_z[order]
            
            for q,p in q_x_list:
                circuit.append("CX", [p, q + num_ancillas])
            
            for q,p in q_z_list:
                circuit.append("CX", [q + num_ancillas, p + len(plaq_d_x)])
            
            circuit.append("TICK")
        
        circuit.append("H", plaq_d_x)
        circuit.append("X_ERROR", full_plaq_L, p_err)
        circuit.append("MR", full_plaq_L)

        # for X mem measure X plaqs
        if self.type == "X":
            for i in range(len(plaq_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i))
        
            circuit.append("Z_ERROR", data_q_list, p_err)
            circuit.append("MX", data_q_list)

            # time to reconstruct each plaquette
            for i in plaq_d_x: 
                
                q_x_list = plaq_d_x[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits + q for q in q_x_list] + [-num_ancillas + anc - num_qubits]
                
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits + d*q) for q in range(d)], 0) # parity of the whole line needs to be the same
        
        # Z mem measure Z plaqs
        if self.type == "Z":
            for i in range(len(plaq_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i + len(plaq_d_x)))
        
            circuit.append("X_ERROR", data_q_list, p_err)
            circuit.append("M", data_q_list)

            # time to reconstruct each plaquette
            for i in plaq_d_z: 
                
                q_z_list = plaq_d_z[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits + q for q in q_z_list] + [-num_ancillas +len(plaq_d_x)+ anc - num_qubits]
                
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits + q) for q in range(d)], 0)
        print(repr(circuit))
        return circuit

    def make_elongated_circuit_from_parity(self, p_err):
        """ 
        create a surface code memory experiment circuit from a parity check matrix

        I think the error type I wanna use is pauli_channel_1(px, py, pz)
        """
        px = 0.5*p_err/(1+self.eta)
        pz = p_err*(self.eta/(1+self.eta))

        # make the circuit
        circuit = stim.Circuit()

        # get the qubit ordering
        plaq_d_x = self.convert_sparse_to_d()
        plaq_d_z = self.convert_sparse_to_d()
        
        # get the qubit ordered properly for each plaquette
        order_d_x = self.check_order_d_elongated()
        order_d_z = self.check_order_d_elongated()
        
        # get the plaquettes that belong to each qubit
        qubit_d_x = self.qubit_to_plaq_d()
        qubit_d_z = self.qubit_to_plaq_d()
        
        # general parameters
        num_ancillas = len(plaq_d_x) + len(plaq_d_z) # total number of plaquettes to initialize
        num_qubits_x = len(qubit_d_x)
        num_qubits_z = len(qubit_d_z)
        
        data_q_x_list = [num_ancillas + q for q in list(qubit_d_x.keys())] # all the x data qubits
        data_q_z_list = [num_ancillas + q for q in list(qubit_d_z.keys())] # all the z data qubits
        data_q_list = data_q_x_list # change this later when wanna do X and Z seperately


        # convention - X plaqs first, then Z plaqs starting with 0
        full_plaq_L = range(num_ancillas)
        
        # reset the ancillas
        circuit.append("R", full_plaq_L)
        circuit.append("X_ERROR", full_plaq_L, px) # add the error to the ancillas
        circuit.append("H", plaq_d_x) # only the X plaqs need H

        # reset the data qubits
        for q in range(len(qubit_d_x)): # go through all the qubits, might need to change when qubit_d_x doesn't have all the qubits
            if type == "X":
                circuit.append("RX", q + num_ancillas)
                circuit.append("Z_ERROR", q + num_ancillas, pz) # add the error to the data qubits
            if type == "Z":
                circuit.append("R", q + num_ancillas)
                circuit.append("X_ERROR", q + num_ancillas, px)
    

        for order in order_d_x: # go through the qubits in order of gates
            q_x_list = order_d_x[order] # (qubit, ancilla)
            # q_z_list = order_d_z[order] # for the idling error on the Z qubits
            
            for q,p in q_x_list:
                circuit.append("CX", [p, q + num_ancillas])
            
            for q,p in q_x_list:
                # CNOT gate errors
                # circuit.append("PAULI_CHANNEL_2", [p, q + num_ancillas], [px, px,pz,px,px**2, px**2, px*pz, px,px**2, px**2, px*pz, pz, pz*px, pz*px, pz**2]) # how do I do the 2-qubit error?
                circuit.append("DEPOLARIZE2", [q + num_ancillas, p + len(plaq_d_x)], p_err)
            # for q,p in q_z_list:
            #     # Idling error on the Z qubits
            #     circuit.append("PAULI_CHANNEL_1", [q + num_ancillas], [px, px,pz]) 
            circuit.append("TICK")


        for order in order_d_z: # go through the qubits in order of gates
            q_z_list = order_d_z[order]
            # q_x_list = order_d_x[order] # for the idling error on the X qubits

            for q,p in q_z_list:
                circuit.append("CX", [q + num_ancillas, p + len(plaq_d_x)])
            for q,p in q_z_list:
                # circuit.append("PAULI_CHANNEL_2", [q + num_ancillas, p + len(plaq_d_x)], [px, px,pz,px,px**2, px**2, px*pz, px,px**2, px**2, px*pz, pz, pz*px, pz*px, pz**2]) # CNOT gate errors
                circuit.append("DEPOLARIZE2", [q + num_ancillas, p + len(plaq_d_x)], p_err) # CNOT gate errors

            # for q,p in q_x_list:
            #     circuit.append("PAULI_CHANNEL_1", [q + num_ancillas], [px, px,pz]) # Idling error on the X qubits
        
        circuit.append("H", plaq_d_x)
        circuit.append("PAULI_CHANNEL_1", full_plaq_L, [px,px,pz])

        circuit.append("X_ERROR", full_plaq_L, px) # add the error to the ancillas
        circuit.append("MR", full_plaq_L)
        circuit.append("X_ERROR", full_plaq_L, px) # add the error to the ancillas
        circuit.append("PAULI_CHANNEL_1", data_q_x_list, [px,px,pz]) # add the error to the ancillas

        # for X mem measure X plaqs
        if type == "X":
            for i in range(len(plaq_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i))

            circuit.append("MX", data_q_x_list)

            # reconstruct each plaquette
            for i in plaq_d_x: 
                q_x_list = plaq_d_x[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits_x + q for q in q_x_list] + [-num_ancillas + anc - num_qubits_x]
                
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits_x + d*q) for q in range(d)], 0) # parity of the whole line needs to be the same
        
        # Z mem measure Z plaqs
        if type == "Z":
            for i in range(len(plaq_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i + len(plaq_d_x)))

            circuit.append("M", data_q_list)

            # time to reconstruct each plaquette
            for i in plaq_d_z: 
                
                q_z_list = plaq_d_z[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits_z + q for q in q_z_list] + [-num_ancillas +len(plaq_d_x)+ anc - num_qubits_z]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_z + q) for q in range(d)], 0)
        return circuit

    def make_clifford_deformed_circuit_from_parity(self, p_err):
        """ Given a parity check matrix pair, generates a STIM circuit and detectors to implement the outlined code.
            Inputs:
                H - (scipy sparse mat) the parity check matrix
                Type - str - type == "X" is the X parity check matrix and produces those stabilizers
                            type == "Z" ' ' 
            Returns: (Stim circuit object) the circuit corresponding to the checks of the specified code

            Note - the X and Z circuits can be seperated in code-capacity model. Circuit level model will
                    require integrating the circuits to avoid hook errors. See fig 3. of this paper: 
                    https://arxiv.org/pdf/1404.3747
        """
        pass


############################################
#
# Functions to test the circuit outputs
#
############################################

def get_num_log_errors(circuit, matching, num_shots):
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

def get_num_log_errors_DEM(circuit, num_shots):
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

def get_log_error_circuit_level(p_list, H_x,H_z, type, eta, d, num_shots):
    """
    Get the logical error rate for a list of p values at the circuit level
    :param p_list: list of p values
    :param H_x: X parity check matrix
    :param H_z: Z parity check matrix
    :param type: type of memory experiment(X or Z)
    :param eta: error rate
    :param d: code distance
    :param num_shots: number of shots to sample
    :return: list of logical error rates
    """
    log_error_L = []
    for p in p_list:
        circuit = make_elongated_circuit_from_parity(H_x,H_z,d, p, eta, type)
        log_errors = get_num_log_errors_DEM(circuit, num_shots)
        log_error_L.append(log_errors/num_shots)
    # print(log_error_L)
    return log_error_L

def get_log_error_p(p_list, H_x,H_z, type, eta, d, num_shots):
    log_error_L = []
    for p in p_list:
        circuit = make_elongated_circuit_from_parity(H_x,H_z, d, p, eta, type)
        matching = Matching.from_stim_circuit(circuit)
        log_errors = get_num_log_errors(circuit, matching, num_shots)
        log_error_L += [log_errors/num_shots]
    return log_error_L







p_list = np.linspace(0,0.5, 30)
d_dict = {5:[], 7:[], 9:[], 11:[]}
num_shots = 100000
l = 3
type_d = {0:"X", 1:"Z"}
type=type_d[1]
eta = 1.67
prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'TOTAL':1}




for d in list(d_dict.keys()):
    compass_code = cc.CompassCode(d=d, l=l)
    H_x, H_z = compass_code.H['X'], compass_code.H['Z']
    # print(H_x) same between files
    log_x, log_z = compass_code.logicals['X'], compass_code.logicals['Z']
    # print('circuit:')
    # my_c = make_elongated_circuit_from_parity(H_x, H_z, d, 0.05, 0.5, type)
    # print(repr(my_c)) same between files
    d_dict[d] = get_log_error_circuit_level(p_list, H_x,H_z, type, eta, d, num_shots)


# matching = Matching.from_stim_circuit(my_c)
# matching.draw()
for d in d_dict:
    plt.plot(p_list*prob_scale[type], d_dict[d], label=f"d={d}")
plt.xlabel("p")
plt.ylabel(f"{[v for k, v in type_d.items() if v != type][0]} Logical Errors ")
plt.legend()
# plt.savefig("l3_eta1.67_zmem.png", dpi=300)
plt.show()