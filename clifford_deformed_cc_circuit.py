import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from scipy import sparse, linalg
import CompassCodes as cc
import stim 
import pandas as pd
import sys

#
# TODO: This code is unfinished. Currently, the surface_code_circuit_scratch file contains the working code that generates a surface code circuit.
# Next:
# (1) modify to work on compass code
# (2) modify to work on XZZX / ZXXZ codes
# (3) modify to work on clifford deformed compass codes

class CDCompassCodeCircuit:
    def __init__(self, d, l, eta, p_params, type):
        self.d = d
        self.l = l
        self.eta = eta
        self.ps = p_params # list: [gate error rate, measurement error rate, idling error rate]
        self.code = cc.CompassCode(d=d, l=l)
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z']
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z']
        self.type = type # str "X" or "Z", indicates the type of memory experiment / which stabilizer you measure

        self.qubit_order_d = self.check_order_d_elongated()
        self.circuit = self.make_elongated_circuit_from_parity()

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
        """ New order based on zigzag pattern in PRA (101)042312, 2020.
            Z stabilizers 
            #    #
            |  / |
            1 2  3 4 .....
            |/   |/
            #    #
            X stabilizers
            #--1--#
                /
               2    .....
             /
            #--3--#

        """
        stab_size = (self.l)*2# the size of the largest stabilizer

        # create the order dictionary to store the qubit ordering for each plaquette
        order_d_x = {}
        order_d_z = {}

        # create the order dictionary to store the qubit ordering for each plaquette
        for row in range(self.H_x.shape[0]):
            order_d_x[row] = []
        for row in range(self.H_z.shape[0]):
            order_d_z[row] = []

        # qubit ordering for Z stabilizers
        for row in range(self.H_z.shape[0]):
            start = self.H_z.indptr[row]
            end = self.H_z.indptr[row+1]
            qubits = sorted(self.H_z.indices[start:end]) # the qubits in the plaquette

            
            for i in range(len(qubits)//2):
                match_qubit_ind = np.where(qubits == (qubits[i] + self.d))[0][0]
                order_d_z[row] += [(qubits[i], row)]
                order_d_z[row] += [(qubits[match_qubit_ind], row)]
        
        
        # qubit ordering for X stabilizers
        for row in range(self.H_x.shape[0]):
            start = self.H_x.indptr[row]
            end = self.H_x.indptr[row+1]
            qubits = sorted(self.H_x.indices[start:end]) # the qubits in the plaquette

            for qubit in qubits:
                order_d_x[row] += [(qubit, row)]
        return order_d_x, order_d_z

    def qubit_to_stab_d(self):  
        """ Given a parity check matrix, returns a dictionary of qubits to plaquettes
            Returns: (dict) qubit to plaquette mapping
        """
        rows_x, cols_x, values = sparse.find(self.H_x)
        rows_z, cols_z, values = sparse.find(self.H_z)
        d_x = {}
        d_z = {}
        for i in range(len(cols_x)):
            q = cols_x[i]
            plaq = rows_x[i]

            if q not in d_x:
                d_x[q] = [plaq]
            else:
                d_x[q] += [plaq]
        
        for i in range(len(cols_z)):
            q = cols_z[i]
            plaq = rows_z[i]

            if q not in d_z:
                d_z[q] = [plaq]
            else:
                d_z[q] += [plaq]

        return d_x, d_z
    
    
    def convert_sparse_to_d(self):

        rows_x, cols_x, values = sparse.find(self.H_x)
        rows_z, cols_z, values = sparse.find(self.H_z)
        d_x = {}
        d_z = {}

        for i in range(len(rows_x)):
            plaq = rows_x[i]
            qubit = cols_x[i]

            if plaq not in d_x:
                d_x[plaq] = [cols_x[i]]
            else:
                d_x[plaq] += [cols_x[i]]
        sorted_d_x = dict(sorted(zip(d_x.keys(),d_x.values())))


        for i in range(len(rows_z)):
            plaq = rows_z[i]
            qubit = cols_z[i]

            if plaq not in d_z:
                d_z[plaq] = [cols_z[i]]
            else:
                d_z[plaq] += [cols_z[i]]
        sorted_d_z = dict(sorted(zip(d_z.keys(),d_z.values())))
        return sorted_d_x, sorted_d_z


    def clifford_deform_parity_mats(self):
        """ Add clifford deformation to parity matrices for elongated compass code with elongation l and
            distance d. Clifford deformations are added according to Julie's model by the transformation XZZXonSq, or transform 2 (i.e. each weight 4
            X stabilizer has 2 H applied to antidiagonal)
            returns: (np array) H_x and H_z with clifford deformations
        """
        H_x, H_z = self.H_x, self.H_z

        # apply transformation row by row
        for row in range(H_x.shape[0]):
            start = self.H_x.indptr[row]
            end = self.H_x.indptr[row+1]
            qubit_inds = self.H_x.indices[start:end] # the qubits in the stabilizer
            


        return H_x, H_z
    


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
    
    def add_meas_round(self, curr_circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, p_i, p_gate, type):
        """
        Add a measurement round to the circuit. Construct the gates with error model 
        for one round of stabilizer construction.
        """
        circuit = curr_circuit
        circuit.append("H", stab_d_x) # only the X stabs need H

        # reset the data qubits
        
        if type == "X":
            circuit.append("RX", [q + num_ancillas for q in range(num_qubits_x)])
            # circuit.append("Z_ERROR", [anc for anc in range(num_ancillas)], p_i) # idling error on the ancillas
            # circuit.append("Z_ERROR", q + num_ancillas, pz) # add the error to the data qubits
        if type == "Z":
            circuit.append("R", [q + num_ancillas for q in range(num_qubits_x)])
            # circuit.append("Z_ERROR", [anc for anc in range(num_ancillas)], p_i) # idling error on the ancillas
            # circuit.append("X_ERROR", q + num_ancillas, px)
    
        # go through each stabilizer in order, X stabilizers first
        for order in order_d_x:
            q_x_list = order_d_x[order] # (qubit, ancilla) in each X stabilizer, qubit is not adjusted for ancilla offset
            q_idling_list = [q for q,_ in q_x_list] # the dummy list for qubits that are idling. All the qubits in the stabilizer idle at some point

            # keep track of the idling qubits outside the stabilizer, including ancillas
            q_inactive_list = [anc for anc in range(num_ancillas) if anc != order]
            for q in range(num_qubits_x):
                if (q, order) not in q_x_list:
                    q_inactive_list.append(q+num_ancillas)

            
            # apply a CX to each qubit in the stabilizer in the correct order
            for q,anc in q_x_list:

                circuit.append("CX", [anc, q + num_ancillas])

                # apply the depolarizing channel to the CX gate
                # circuit.append("DEPOLARIZE2", [anc, q+num_ancillas], p_gate)

                # apply idling errors to the qubits in the stabilizer without CX
                # for other_q in q_idling_list:
                #     if other_q != q:
                        # circuit.append("Z_ERROR", [other_q + num_ancillas], p_i) # Idling error on the X qubits 
                # circuit.append("Z_ERROR", q_inactive_list, p_i) # Idling error on the ancillas and qubits outside the stabilizer

            circuit.append("TICK")

        # now do the Z stabilizers
        for order in order_d_z: 
            q_z_list = order_d_z[order] # (qubit, ancilla) in each stabilizer, not offset for x ancillas for the z ancillas, or ancillas for the data qubits
            q_idling_list = [q for q,_ in q_z_list] # the dummy list for qubits that are idling


            # keep track of the idling qubits outside the stabilizer
            q_inactive_list = [anc for anc in range(num_ancillas) if anc != (order + len(stab_d_x))]
            for q in range(num_qubits_z):
                if (q, order) not in q_z_list:
                    q_inactive_list.append(q+num_ancillas)

            # apply a CX to each qubit in the stabilizer in the correct order
            for q,anc in q_z_list:
                circuit.append("CX", [q + num_ancillas, anc + len(stab_d_x)]) # ancillas are shifted to account for X stabs

                # circuit.append("DEPOLARIZE2", [q + num_ancillas, anc + len(stab_d_x)], p_gate) # CNOT gate errors

                # apply idling errors to the qubits in the stabilizer without CX
                # for other_q in q_idling_list:
                #     if other_q != q:
                #         circuit.append("Z_ERROR", [other_q + num_ancillas], p_i) # Idling error on the X qubits
                # circuit.append("Z_ERROR", q_inactive_list, p_i) # Idling error on the ancillas and qubits outside the stabilizer
            
            circuit.append("TICK")
        
        circuit.append("H", stab_d_x)
        # circuit.append("Z_ERROR", [q for q in range(len(stab_d_x), num_ancillas + num_qubits_x)], p_i) # idling error on the ancillas
        return circuit

    

    def make_elongated_circuit_from_parity(self):
        """ 
        create a surface code memory experiment circuit from a parity check matrix
        Inputs:
                circuit - (stim.Circuit) the circuit to add noise to
                p_gate - (float) the probability of a gate error
                p_meas - (float) the probability of a measurement error
                p_i - (float) the probability of an idling error
            Returns: (stim.Circuit) the circuit with noise added

            The error model is the biased noise model from the paper: PRA (101)042312, 2020
            - 2-qubit gates are followed by 2-qubit depolarizing channel with p = p_gate (x)
            - measurement outcomes are followed by a bit flip with probability p_meas (x)
            - idling qubits are followed by a dephasing channel with probability p_i (x)

        """
        p_gate = self.ps[0] # gate error on two-qubit gates
        p_meas = self.ps[1] # measurement error
        p_i = self.ps[2] # idling error, to be scanned over 
        
        px = 0.5*p_i/(1+self.eta)
        pz = p_i*(self.eta/(1+self.eta))

        # make the circuit
        circuit = stim.Circuit()

        # get the qubit ordering
        stab_d_x,stab_d_z = self.convert_sparse_to_d()
        
        # get the qubit ordered properly for each stabilizer
        order_d_x, order_d_z = self.check_order_d_elongated()
        
        # get the stabilizer that belong to each qubit
        qubit_d_x,qubit_d_z = self.qubit_to_stab_d()
        
        # general parameters
        num_ancillas = len(stab_d_x) + len(stab_d_z) # total number of stabilizer to initialize
        num_qubits_x = len(qubit_d_x)
        num_qubits_z = len(qubit_d_z)

        # print(num_ancillas, num_qubits_x, num_qubits_z)
        # print(self.H_x.shape, self.H_z.shape)
        # print(stab_d_x, stab_d_z)
        
        data_q_x_list = [num_ancillas + q for q in list(qubit_d_x.keys())] # all the x data qubits
        data_q_z_list = [num_ancillas + q for q in list(qubit_d_z.keys())] # all the z data qubits
        data_q_list = data_q_x_list # change this later when wanna do X and Z seperately


        # convention - X stabs first, then Z stabs starting with 0
        full_stab_L = range(num_ancillas)
        
        # reset the ancillas
        circuit.append("R", full_stab_L)
        # circuit.append("X_ERROR", full_stab_L, px) # add the error to the ancillas


        # start the for loop to repeat for d rounds

        # Round 0 - t=0 measurements

        circuit = self.add_meas_round(circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, p_i, p_gate, self.type)

        # idling errors on the data qubits
        # circuit.append("Z_ERROR", data_q_z_list, p_i)
        # circuit.append("X_ERROR", full_stab_L, p_meas) # add the error to the ancillas
        circuit.append("X_ERROR", full_stab_L, p_i) # for phenom only
        circuit.append("MR", full_stab_L)
        circuit.append("X_ERROR", full_stab_L,p_i) # add the error to the ancillas
        # circuit.append("Z_ERROR", data_q_z_list, p_i)


        # initialize the t=0 detectors for the X or Z stabilizers
        if self.type == "X":
            for i in range(len(stab_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i))
        elif self.type == "Z":
            for i in range(len(stab_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i + len(stab_d_x)))
        
        circuit.append("TICK") # add a tick to the circuit to mark the end of the t=0 measurements
        
        loop_circuit = stim.Circuit() # create a loop circuit to repeat the following for d-1 rounds
        # All other d rounds - t>0 measurements
        # circuit += "REPEAT %d {\n" # repeat the following for d-1 rounds
        # add a measurement round
        loop_circuit = self.add_meas_round(loop_circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, p_i, p_gate, self.type)

        # idling errors on the data qubits, measure the ancillas, bit flip errors on measurements
        # loop_circuit.append("Z_ERROR", data_q_z_list, p_i)
        loop_circuit.append("X_ERROR", full_stab_L, p_i)
        loop_circuit.append("MR", full_stab_L)
        loop_circuit.append("X_ERROR", full_stab_L, p_i) # add the error to the ancillas
        # loop_circuit.append("X_ERROR", full_stab_L, p_meas) # add the error to the ancillas
        # loop_circuit.append("PAULI_CHANNEL_1", data_q_z_list, [0,0,p_i])

        # timelike detectors for the X or Z stabilizers
        if self.type == "X":
            for i in range(len(stab_d_x)):
                loop_circuit.append("DETECTOR", [stim.target_rec(-num_ancillas + i), stim.target_rec(-2*num_ancillas + i)]) # anc round d tied to anc round d=0
        elif self.type == "Z":
            for i in range(len(stab_d_z)):
                loop_circuit.append("DETECTOR", [stim.target_rec(-num_ancillas + i + len(stab_d_x)), stim.target_rec(-2*num_ancillas + i + len(stab_d_x))]) # anc round d tied to anc round d=0
        loop_circuit.append("TICK") # add a tick to the circuit to mark the end of the t>0 iteration
        
        # repeat the loop circuit d-1 times - circuit level only
        circuit.append(stim.CircuitRepeatBlock(repeat_count=self.d-1, body=loop_circuit))# end the repeat block

        # reconstruct the stabilizers and measure the data qubits
        # for X mem measure X stabs
        if self.type == "X":
            # measure all the data qubits in the X stabilizers
            circuit.append("X_ERROR", data_q_x_list, p_i) # add the error to the data qubits
            circuit.append("MX", data_q_x_list)
            circuit.append("X_ERROR", data_q_x_list, p_i)

            # reconstruct each X stabilizer with a detector
            for anc in stab_d_x: 
                q_x_list = stab_d_x[anc] # get the qubits in the stab
                detector_list =  [-num_qubits_x + q for q in q_x_list] + [-num_ancillas + anc - num_qubits_x]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
            
            
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits_x + self.d*q) for q in range(self.d)], 0) # parity of the whole line needs to be the same
        
        # Z mem measure Z stabs
        if self.type == "Z":
            # measure all the data qubits in the Z stabilizers
            circuit.append("X_ERROR", data_q_z_list, p_i) # add the error to the data qubits
            circuit.append("M", data_q_list)
            circuit.append("X_ERROR", data_q_z_list, p_i)

            # reconstruct each stabilizer with a detector
            for anc in stab_d_z: 
                
                q_z_list = stab_d_z[anc] # get the qubits in the stab
                detector_list =  [-num_qubits_z + q for q in q_z_list] + [-num_ancillas +len(stab_d_x)+ anc - num_qubits_z]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_z + q) for q in range(self.d)], 0)
        return circuit

    def make_clifford_deformed_circuit_from_parity(self):
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
        circuit = stim.Circuit()
        


        return circuit


# for d rounds
# 3 sets of detectors:
# before the for loop, measure the ancillas that I want to track (ie X mem measure X stab ancillas)
# during the for loop, for each round, make a detector between each stabilizer and itself at time 0. That is, measure the ancilla at round d and make a detector with the same ancilla outside the for loop
# after the for loop, reconstruct the stabilizers in mem of interest, and measure the data qubits in each stabilizer (ie tie an anc for an X stab in the last iteration of the for loop to the data qubits in the X stab)




p_list = np.linspace(0,0.5, 30)
d_dict = {5:[], 7:[], 9:[], 11:[]}
num_shots = 100000
l = 2
type_d = {0:"X", 1:"Z"}
type=type_d[0]
eta = 1.67
prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'TOTAL':1}



# circuit = CDCompassCodeCircuit(5, l, eta, [0.003, 0.001, 0.05], type)



# order_d = circuit.check_order_d_elongated() # looks mostly good
# print(circuit.H_x)
# print(order_d)
# for d in list(d_dict.keys()):
#     circuit = CDCompassCodeCircuit(d, l, eta, 0.05, type)
#     # print('circuit:')
    # my_c = make_elongated_circuit_from_parity(H_x, H_z, d, 0.05, 0.5, type)
    # print(repr(my_c)) same between files
    # d_dict[d] = circuit.get_log_error_circuit_level(p_list, H_x,H_z, type, eta, d, num_shots)


# matching = Matching.from_stim_circuit(my_c)
# matching.draw()
# for d in d_dict:
#     plt.plot(p_list*prob_scale[type], d_dict[d], label=f"d={d}")
# plt.xlabel("p")
# plt.ylabel(f"{[v for k, v in type_d.items() if v != type][0]} Logical Errors ")
# plt.legend()
# # plt.savefig("l3_eta1.67_zmem.png", dpi=300)
# plt.show()