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
import collections
from datetime import datetime
import sys
import glob
from scipy.optimize import curve_fit
import clifford_deformed_cc_circuit as cc_circuit
import itertools
import stim
# from lmfit import Minimizer, Parameters, report_fit


##############################################
#
# CorrelatedDecoder class
#
##############################################

class CorrelatedDecoder:
    def __init__(self, eta, d, l, corr_type, mem_type="X"):
        self.eta = eta # the noise bias
        self.d = d # the distance of the compass code
        self.l = l # the elongation parameter
        self.corr_type = corr_type # the type of correlation for decoder (directional)
        self.mem_type = mem_type
        self.edge_type_d = {} # dictionary of the edge types for each detector. Empty until populated by running method to populate. Type 0(1) use pauli X(Z) measurements

        self.code = cc.CompassCode(d=self.d, l=self.l)
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z'] # parity check matrices from compass code class
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z'] # logical operators from compass code class

    def bernoulli_prob(self, old_prob, p):
        """ Given an old probability and a new error probability, return the updated probability
            according to the bernoulli formula
        """
        new_prob = old_prob*(1-p) + p*(1 - old_prob)
        return new_prob  

    def get_dB_scaling(self, matching):
        edge = next(iter(matching.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        decibels_per_w = -np.log10(edge_p / (1 - edge_p)) * 10 / edge_w 
        return decibels_per_w

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
    
    def test_error(self, error_x, error_z):

        M_x = Matching.from_check_matrix(self.H_x)
        M_z = Matching.from_check_matrix(self.H_z)
        
        syndrome_x, syndrome_z = error_x @ self.H_z.T % 2, error_z @ self.H_x.T % 2
        print(f"syndrome for X errors {syndrome_x}")
        print(f"syndrome for Z errors {syndrome_z}")

        correction_x = M_z.decode(syndrome_x)
        correction_z = M_x.decode(syndrome_z)

        print(f"correction for X errors {correction_x}")
        print(f"correction for Z errors {correction_z}")

        
    
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
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_x @ self.H_z.T % 2
        correction_x = M_z.decode_batch(syndrome_x)
        num_errors_x = np.sum((correction_x + err_vec_x) @ self.log_z % 2)
        
        # Syndrome for Z errors and decoding
        syndrome_z = err_vec_z @ self.H_x.T % 2
        correction_z = M_x.decode_batch(syndrome_z)
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
    
    ########################################################################
    #
    # Circuit level correlated decoding functions
    #
    ########################################################################



    #
    # Graph labelling / edge tracking
    #

    def probability_edge_mapping(self, edge_dict):
        """ Maps the probabilities to the corresponding edge weight in the matching graph. Takes into
            account the 'type' of qubit, whether it is clifford deformed or not. CURRENTLY DOES NOT TAKE INTO 
            ACCOUNT THE TYPE OF QUBIT - NOT SURE WHAT THIS MEANS / HOW TO ACCOUNT CD
        """
        weights_dict = {}

        for edge_1 in edge_dict:

            adjacent_edge_dict = edge_dict.get(edge_1, {})

            # populate weight dictionary 
            for edge_2 in adjacent_edge_dict:

                p = edge_dict.get(edge_1, {}).get(edge_2,0)
                weight = np.log((1-p)/p)
                weights_dict.setdefault(edge_1, {})[edge_2] = weight
        
        return weights_dict
    
    def get_qubit_in_edge(self, edge_type, stab1, stab2) -> np.ndarray:
        """
        Return the qubits involved in the edge connecting two stabilizers.
        If the stabilizer is connected to a boundary (stab == -1),
        return empty for that side.
        """

        n_qubits = self.H_x.shape[1]   # <-- number of columns = number of qubits
        qubits_stab1 = sparse.csr_matrix(np.zeros(n_qubits, dtype=int))
        qubits_stab2 = sparse.csr_matrix(np.zeros(n_qubits, dtype=int))

        if edge_type == 1:
            # Z stabilizers
            if stab1 != -1:
                qubits_stab1 = self.H_z.getrow(stab1 - self.H_x.shape[0])

            if stab2 != -1:
                qubits_stab2 = self.H_z.getrow(stab2 - self.H_x.shape[0])

        elif edge_type == 0:
            # X stabilizers
            if stab1 != -1:
                qubits_stab1 = self.H_x.getrow(stab1)

            if stab2 != -1:
                qubits_stab2= self.H_x.getrow(stab2)

        qubits_in_edge = qubits_stab1.multiply(qubits_stab2).indices

        return qubits_in_edge

    
    def get_edge_type_from_detector(self, edge, mem_type, CD_data_transform) -> int:
        """
        Returns the edge type (0 or 1) for a given edge in the DEM. Type 0(1) connect Pauli X(Z) measurements.  
        """
        d1 = edge[0]
        d2 = edge[1]
        stab1 = self.get_stab_from_detector(d1, mem_type)
        stab2 = self.get_stab_from_detector(d2, mem_type)

        num_stabs_r1 = self.H_x.shape[0] if mem_type == "X" else self.H_z.shape[0]
        edge_type = 0

        if stab1 >= self.H_x.shape[0] and stab2 >= self.H_x.shape[0]: # Z type edge in SC
            edge_type = 1
        elif stab1 < self.H_x.shape[0] and stab2 < self.H_x.shape[0]:
            edge_type = 0
        elif stab1 == -1 or stab2 ==-1:
            edge_type = 0 if max(stab1,stab2) < self.H_x.shape[0] else 1
        else:
            edge_type = 2 # edge between X and Z types ... don't touch this - directly from DEM during perfect round
        
        # apply the deformation if necessary
        qubit_in_edge = self.get_qubit_in_edge(edge_type, stab1, stab2)
        if qubit_in_edge.size == 0:
            CD_applied = 0
        elif abs(d1 - d2) >= num_stabs_r1:
            CD_applied = 0
        else:
            CD_applied = CD_data_transform[qubit_in_edge[0]]

        
        if CD_applied == 2 and edge_type != 2: # if there is a Hadamard on that qubit, swap the edge type
            edge_type = (edge_type + 1)%2
        
        return edge_type
    
    def get_stab_from_detector(self, detector, mem_type) -> int:
        """
        Returns the stabilizer index for a given detector in the DEM. This is used to determine which stabilizer measurement type (X or Z) is associated with a given detector.

        Inputs:
        detector - (integer) the value of the detector of question
        
        Outputs:
        stab_index - (integer) the index of the stabilizer included in the detector. The full stabilizer list includes X and Z types.
        """
        stab_index=0
        curr_det_index = detector

        if detector == -1:
            return -1

        # should be d*(Hx.shape[0] + Hz.shape[0]) detectors

        # X detectors are always the first half
        if mem_type == "X":
            if detector < self.H_x.shape[0]: # the detector is in the first layer and is an X detector for sure but whatever
                stab_index = detector
            elif detector > self.H_x.shape[0] + (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]): # last layer of checks
                stab_index = detector- (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]) - self.H_x.shape[0]
            else:
                curr_det_index -= self.H_x.shape[0]
                stab_index = curr_det_index % (self.H_x.shape[0] + self.H_z.shape[0])
        else: # Z detectors
            if detector < self.H_z.shape[0]: 
                stab_index = detector + self.H_x.shape[0] # Z stabs are offset by X ones, check X first
            elif detector >= self.H_z.shape[0] + (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]):
                stab_index = detector - (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]) - self.H_z.shape[0] + self.H_x.shape[0]
            else:
                curr_det_index -= self.H_z.shape[0]
                stab_index = curr_det_index%(self.H_x.shape[0] + self.H_z.shape[0])
        
        return stab_index
    
    def get_LB_RB_nodes(self, DEM):
        """ Get a list of the LB and RB on the code (closed boundary for each stabilizer type)
            Inputs - (detector error model) the model representing errors in system of choice
            Outputs - (list)s the lists of the X measurement L/R stabilizers, and the Z measurement
                        T/B stabilizers. Orthogonal to that logical type
        """
        xlb_nodes = [] # the detectors that correspond to left X stabilizers
        xrb_nodes = [] # '' right X stabilizers
        ztb_nodes = [] # '' top Z stabilizers
        zbb_nodes = [] # '' bottom Z stabilizers


        # for detector in DEM ...
        detectors = DEM.num_detectors

        # get the stab index from the detector
        for d in range(detectors):
            stab_index = self.get_stab_from_detector(d, self.mem_type)
            # print("stab and d",stab_index, d)

            # Assign X left/right stabilizers
            if stab_index < self.H_x.shape[0]: # it is an X detector

                qubits_in_stab = sorted(self.H_x.getrow(stab_index).indices)
                # print(qubits_in_stab)

                if qubits_in_stab[0] % self.d == 0: # on the left
                    # print(qubits_in_stab[0] % self.d)
                    xlb_nodes += [d]
                elif (qubits_in_stab[-1] +1)% self.d == 0: # on the right 
                    xrb_nodes += [d]
                else:
                    pass
            # Check if Z are top/bottom
            else: # now these are Z detectors
                qubits_in_stab = sorted(self.H_z.getrow(stab_index - self.H_x.shape[0]).indices)
                # print(qubits_in_stab[0])
                if qubits_in_stab[0] < self.d:
                    ztb_nodes += [d]
                elif qubits_in_stab[-1] >= (self.d**2-self.d):
                    zbb_nodes += [d]

        return xlb_nodes,xrb_nodes,ztb_nodes,zbb_nodes


    def get_edge_type_d(self, dem, mem_type, CD_type) -> dict:
        """
        Returns the dictionary mapping edges in the DEM to stabilizer measurement types. Updates the edge_type_d attribute of the class.
        The dictionary is for marginal edges only: hyperedges are assumed decomposed.
        Inputs:
        dem - (stim.DetectorErrorModel) the dem noise model used for the code
        Outputs:
        edge_type_d - (dict) a dictionary mapping edges in the DEM to stabilizer measurement types. Type 0(1) use pauli X(Z) measurements
            eg. {(0,-1):0, (2,4):1, (3,5):0, ...}
        """
        if CD_type != "SC":
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special=CD_type, ell=self.l, size=self.d)
        else:
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special="I", ell=self.l, size=self.d)

        for inst in dem:
            if inst.type == "error":
                decomposed_inst = self.decompose_dem_instruction_stim(inst)

                for edge in decomposed_inst["detectors"]:
                    # print(edge)
                    if tuple(sorted(edge)) in self.edge_type_d:
                        pass
                    else:
                        self.edge_type_d[tuple(sorted(edge))] = self.get_edge_type_from_detector(edge, mem_type, CD_data_transform)
                    
        return self.edge_type_d
    

    #
    # Complementary Gap
    #

    def get_complementary_gap(self,dem,syndrome,obs_flips):
        '''
        Credit: Eva Takou for code backbone. Minor style changes have been made. The original function
        calculates the complementary gap (MWPM soft information) in the style of arxiv:2312.04522

        Inputs: 
        matching: the pymatching graph
        syndrome: the detector syndrome
        obs_flips: Z(X) logical flipped in X(Z) memory
        b1_nodes: the X(Z) detector nodes to the left(top) boundary for X(Z) memory (list of ints)
        b2_nodes: the X(Z) detector nodes to the right(bottom) boundary for X(Z) memory (list of ints)

        Outputs:
        Gap:                complementary gap
        Signed_Gap:         signed complementary gap
        gap_conditioned_PL: gap conditioned logical error rate
        
        
        '''    
        
        num_shots = np.shape(syndrome)[0]
        comp_matching = Matching()
        matching = Matching.from_detector_error_model(dem)

        xlb_nodes, xrb_nodes, ztb_nodes, zbb_nodes = self.get_LB_RB_nodes(dem)

        if self.mem_type == "X":
            b1_nodes = xlb_nodes
            b2_nodes = xrb_nodes
        else:
            b1_nodes = ztb_nodes
            b2_nodes = zbb_nodes


        b1 = max(b2_nodes)+1
        b2 = b1+1
        
        for edge in matching.edges():
            node1 = edge[0]
            node2 = edge[1]


            # when the edge is not a boundary add to the graph normally
            if node2 is not None:
                
                comp_matching.add_edge(node1=node1,node2=node2,
                                fault_ids = edge[2]['fault_ids'],
                                weight=edge[2]['weight'],
                                error_probability=edge[2]['error_probability'])
            
            # if the edge is a boundary edge 
            else: 
                if node1 in b1_nodes: # match to the left/top node
                    node2 = b1 
                if node1 in b2_nodes: # match to the right/bottom node
                    node2 = b2 

                # if the stabilizer is not of the memory type, keep to normal boundaries
                if node2 is None:
                    comp_matching.add_boundary_edge(node=node1,
                                                    fault_ids = edge[2]['fault_ids'],
                                                    weight=edge[2]['weight'],
                                                    error_probability=edge[2]['error_probability'])

                else:
                    comp_matching.add_edge(node1=node1,node2=node2,
                                    fault_ids = edge[2]['fault_ids'],
                                    weight=edge[2]['weight'],
                                    error_probability=edge[2]['error_probability'])            
                
        
        comp_matching.set_boundary_nodes({b2})     
                
        # decode to obtain the original matching
        pred_reg, W_reg = matching.decode_batch(syndrome,return_weights=True) #This is the regular matching


        # don't fire the b1
        new_array = np.zeros((num_shots,1),dtype=int)
        det0      = np.hstack((syndrome,new_array))
        
        # do fire b1
        new_array = np.ones((num_shots,1),dtype=int)
        det1      = np.hstack((syndrome,new_array))

        # the I_L / ERR_L complementary matchings. Return the total weights of the solutions for each shot
        pred0, W0 = comp_matching.decode_batch(det0,return_weights=True)
        pred1, W1 = comp_matching.decode_batch(det1,return_weights=True)


        # scale by edge weight, get dB. Why do we do this? also do we assume all edges normalized by the weight of the first
        edge = next(iter(matching.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        decibels_per_w = -np.log10(edge_p / (1 - edge_p)) * 10 / edge_w                

        # Unsigned gap
        Gap = []
        for k in range(num_shots):
            if W1[k]<W0[k]:
                Gap.append( (W0[k]-W1[k]) * decibels_per_w)
            else:
                Gap.append( (W1[k]-W0[k]) * decibels_per_w)     

        
        # signed gap - negative indicates MWPM failed
        Signed_Gap = []
        W_min = np.zeros(W_reg.shape)
        W_comp = np.zeros(W_reg.shape)
        pred_min = np.zeros(pred0.shape)


        for k in range(num_shots):
            if W_reg[k] == W0[k]: 
                W_min[k] = W0[k]
                pred_min[k] = pred0[k]
                W_comp[k] = W1[k]
            elif W_reg[k] == W1[k]:
                W_min[k] = W1[k]
                pred_min[k] = pred1[k]
                W_comp[k] = W0[k]

            if pred_min[k]==obs_flips[k]: 
                Signed_Gap.append( (W_comp[k]-W_min[k]) * decibels_per_w) 
            else:
                Signed_Gap.append( (W_min[k]-W_comp[k]) * decibels_per_w) 


        errors = np.any(pred_reg != obs_flips, axis=1)

        # Classify all shots by their error + gap.
        custom_counts = collections.Counter()
        Gap  = np.round(Gap).astype(dtype=np.int64)
        for k in range(len(Gap)):
            g = Gap[k]
            key = f'E{g}' if errors[k] else f'C{g}'
            custom_counts[key] += 1/num_shots

        # P_L(e | g) = E_g / (E_g + C_g)
 
        gap_conditioned_PL = {}

        # collect all gap values that appear
        gaps = set()
        for key in custom_counts:
            gaps.add(int(key[1:]))

        for g in gaps:
            E = custom_counts.get(f'E{g}', 0.0)
            C = custom_counts.get(f'C{g}', 0.0)

            if E + C > 0:
                gap_conditioned_PL[g] = E / (E + C)
            else:
                gap_conditioned_PL[g] = np.nan    


        return Gap,Signed_Gap,gap_conditioned_PL
    
    def get_complementary_correction(self, dem, syndrome, observable_flip, input_matching=None, return_predictions=False):
        """ For one shot at a time, get the unsigned gap, the matching and the complementary matching for one dem

            :param dem: (stim.DetectorErrorModel) the input detector error model to be used in matching
            :param syndrome: (numpy array) the detectors flipped in the experiment
            :param observable_flip: (numpy array) whether a logical observable was flipped
            :param matching: (Matching matching object) if you want to directly feed in a matching graph, use this instead of dem
            :param return_predictions: (bool) include the prediction in the return value

            :return unsigned_gap: (array) decoder confidence from comparing two matchings
            :return matching_correction: (array) the edges included in the min weight solution
            :return comp_matching_correction: (array) the edges in the solution to complementary solution
            :return pred_min: (bool) whether the min weight decoder solution flipped a logical
            :return pred_picked: (int) whether the solution is connected to boundaries(no boundaries) - 1(0)
        """
        
        comp_matching = Matching()
        if input_matching is None:
            matching = Matching.from_detector_error_model(dem)
        else:
            matching = input_matching

        syndrome = syndrome.reshape(1,syndrome.shape[0]) # I hope this is doing the right thing not sure it is
        xlb_nodes, xrb_nodes, ztb_nodes, zbb_nodes = self.get_LB_RB_nodes(dem)

        if self.mem_type == "X":
            b1_nodes = xlb_nodes
            b2_nodes = xrb_nodes
        else:
            b1_nodes = ztb_nodes
            b2_nodes = zbb_nodes


        b1 = max(b2_nodes)+1
        b2 = b1+1
        
        for edge in matching.edges():
            node1 = edge[0]
            node2 = edge[1]


            # when the edge is not a boundary add to the graph normally
            if node2 is not None:
                
                comp_matching.add_edge(node1=node1,node2=node2,
                                fault_ids = edge[2]['fault_ids'],
                                weight=edge[2]['weight'],
                                error_probability=edge[2]['error_probability'])
            
            # if the edge is a boundary edge 
            else: 
                if node1 in b1_nodes: # match to the left/top node
                    node2 = b1 
                if node1 in b2_nodes: # match to the right/bottom node
                    node2 = b2 

                # if the stabilizer is not of the memory type, keep to normal boundaries
                if node2 is None:
                    comp_matching.add_boundary_edge(node=node1,
                                                    fault_ids = edge[2]['fault_ids'],
                                                    weight=edge[2]['weight'],
                                                    error_probability=edge[2]['error_probability'])

                else:
                    comp_matching.add_edge(node1=node1,node2=node2,
                                    fault_ids = edge[2]['fault_ids'],
                                    weight=edge[2]['weight'],
                                    error_probability=edge[2]['error_probability'])            
                
        
        comp_matching.set_boundary_nodes({b2})     
                
        # decode to obtain the original matching
        pred_reg, W_reg = matching.decode(syndrome,return_weight=True) #This is the regular matching


        # don't fire the b1 - I_L coset
        new_array = np.zeros((1,1),dtype=int)
        det0      = np.hstack((syndrome,new_array))
        
        # do fire b1 - Z/X_L coset
        new_array = np.ones((1,1),dtype=int)
        det1      = np.hstack((syndrome,new_array))

        # pred0 crosses logical even number of times, pred1 crosses odd number. Return the total weights of the solutions for each shot
        pred0, W0 = comp_matching.decode(det0,return_weight=True)
        pred1, W1 = comp_matching.decode(det1,return_weight=True)

        edges_in_pred0 = np.array(comp_matching.decode_to_edges_array(det0))
        edges_in_pred1 = np.array(comp_matching.decode_to_edges_array(det1))


        # signed gap
        if W_reg == W0: # MWPM picked pred0 solution
            pred_picked = 0
            W_min = W0
            pred_min = pred0
            W_comp = W1
            edges_in_correction = np.where(np.logical_or((edges_in_pred0 == b1) ,(edges_in_pred0 == b2)), -1, edges_in_pred0)
            edges_in_comp_correction = np.where(np.logical_or((edges_in_pred1 == b1), (edges_in_pred1 == b2)), -1, edges_in_pred1)
        else: # MWPM picked pred 1 solution 
            pred_picked = 1
            W_min = W1
            pred_min = pred1
            W_comp = W0
            edges_in_correction = np.where(np.logical_or((edges_in_pred1 == b1), (edges_in_pred1 == b2)), -1, edges_in_pred1)
            edges_in_comp_correction = np.where(np.logical_or((edges_in_pred0 == b1), (edges_in_pred0 == b2)), -1, edges_in_pred0)
        

        # if pred_min == observable_flip: # MWPM was successful 
        #     signed_gap = W_comp - W_min
        # else:
        #     signed_gap = W_min - W_comp

        unsigned_gap = W_comp - W_min


        if return_predictions:
            return unsigned_gap, edges_in_correction, edges_in_comp_correction, pred_min, pred_picked
        else:
            return unsigned_gap, edges_in_correction, edges_in_comp_correction



    #
    # Hyperedge decomposition (only decompose_dem_instruction_stim used)
    #


    def decompose_dem_instruction_stim_auto(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses STIM's decompose_errors to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to odd cardinality hyperedges. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.

            eg. error(p) D0 ^ D1 L0 -> {p: [(0, 1)]}
                error(p) D0 D2 ^ D1 -> {p: [(0, 2), (1, "BOUNDARY")]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary with the probability as the key and a list of edges as the value.
        """
        # get the edge probability and detectors for an instruction
        prob_err = inst.args_copy()[0]
        targets = np.array(inst.targets_copy())
        decomp_inst = {prob_err: []}

        
        seperator_indices = np.where([target.is_separator() for target in targets])[0]
        split_indices = seperator_indices + 1
        edges = np.split(targets, split_indices)
        edges = [[e.val for e in edge if e.is_relative_detector_id()] for edge in edges]
        total_num_detectors = sum([len(edge) for edge in edges])
        if total_num_detectors > 2:
            for edge in edges:
                if len(edge) % 2 == 1 and len(edges) > 1:
                    edge.append("BOUNDARY")
        
        # Convert edges to list of tuples
        if total_num_detectors <= 2:
            # Flatten and group into one tuple if <= 2 detectors total
            flattened = [e for edge in edges for e in edge]
            edges = [tuple(sorted(flattened, key=lambda x: (isinstance(x, str), x)))]
        else:
            edges = [tuple(sorted(edge, key=lambda x: (isinstance(x,str), x))) for edge in edges]

        # Store result
        decomp_inst[prob_err] = edges

        return decomp_inst
    
    def decompose_dem_instruction_stim(self, inst):
        """
        Decomposes a stim DEM instruction into pairwise detector edges and assigns observables
        to the edges based on which sub-block (separated by `^`) the observable appeared in. Use
        stim DEM instruction decomposition from decompose_errros=True to choose hyperedge 
        decomposition

        Example:
            error(p) D0 D1^D2 L0 -> {p:p, detectors: [(0, -1), (2, -1)], observables: [None, 0]}
            error(p) D0 D1 L0^D2 -> {p:p, detectors: [(0, 1), (-1, 2)], observables: [0, None]}
            error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (-1, 3)], observables:[None, None]}
            error(p) D0 -> {p: p, detectors: [(-1, 0)], observables: [None]} 

        Returns:
            {
                'p': float,
                'detectors': List[Tuple[int, int]],
                'observables': List[Optional[int]],
            }
        """
        targets = list(inst.targets_copy())
        p = inst.args_copy()[0]

        blocks = []  # Each block is a list of targets between separators (^)
        current_block = []

        for t in targets:
            if t.is_separator():
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(t)

        if current_block:
            blocks.append(current_block)

        detector_edges = []
        edge_observables = []

        for block in blocks:
            dets = []
            obs = []

            for t in block:
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)

            # Handle detectors → edges
            if len(dets) == 0:
                continue  # no detector => no edge
            elif len(dets) == 1:
                edge = (-1, dets[0])  # boundary edge
                detector_edges.append(edge)
                edge_observables.append(obs[0] if obs else None)
            else:
                # Decompose pairwise through chain
                for i in range(len(dets) - 1):
                    edge = tuple(sorted((dets[i], dets[i+1])))
                    detector_edges.append(edge)
                    edge_observables.append(obs[0] if obs else None)

        return {
            "p": p,
            "detectors": detector_edges,
            "observables": edge_observables
        }



    def decompose_dem_instruction_pairwise(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses pairwise decomposition to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to edges with one detector, boundary node value is -1. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.


            eg. error(p) D0 D1 L0 -> {p: p, detectors: [(0, 1)], observables: [0]}
                error(p) D0 -> {p: p, detectors: [(-1, 0)], observables: []} single detector error gets boundary edge
                error(p) D0 D2 D1 -> {p:p, detectors: [(0, 2), (2, 1)], observables: []}. 
                error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[]} We choose to ignore ^. If we treated the ^ as already decomposing, we would get [(0,2), (3,-1)]
                error(p) D0 D2 D3 L0 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[0]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary recording the probability of the error for that DEM instruction, the edges included in the
            decomposition, and the logical observables included.
        """
        # get the edge probability and detectors for an instruction
        targets = list(inst.targets_copy())
        decomp_inst = {"p": inst.args_copy()[0], "detectors": [], "observables": []}

        # separate detectors, logical observables, and separators
        for t in targets:
            if t.is_separator():
                continue
            elif t.is_logical_observable_id():
                # logical observable: L#
                decomp_inst["observables"].append(t.val)
            elif t.is_relative_detector_id():
                # detector: D#
                decomp_inst["detectors"].append(t.val)
        
        total_num_detectors = len(decomp_inst["detectors"])

        # iterate through array and make pairwise edge tuples with probability prob_err
        detectors = decomp_inst["detectors"]
        edges = []

        if total_num_detectors == 1:
            edges = [(-1, detectors[0])] # include a boundary edge
        
        else: # pairwise decompose
            for i in range(total_num_detectors-1):
                edges.append(tuple(sorted([detectors[i], detectors[i+1]])))
        
        # store result
        decomp_inst["detectors"] = edges
        return decomp_inst
    
    def decompose_dem_instruction_star(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses star decomposition to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to edges with one detector, boundary node value is -1. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.
            PASS IN DEM with DECOMPOSE_ERRORS=FALSE - talk to ken about this


            eg. error(p) D0 D1 L0 -> {p: p, detectors: [(0, 1)], observables: [0]}
                error(p) D0 -> {p: p, detectors: [(0, -1)], observables: []} single detector error gets boundary edge
                error(p) D0 D2 D1 -> {p:p, detectors: [(0, 2), (0, 1)], observables: []}. 
                error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[]} We choose to ignore ^. If we treated the ^ as already decomposing, we would get [(0,2), (3,-1)]
                error(p) D0 D2 D3 L0 -> {p:p, detectors: [(0, 2), (0, 3)], observables:[0]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary recording the probability of the error for that DEM instruction, the edges included in the
            decomposition, and the logical observables included.
        """
        # get the edge probability and detectors for an instruction
        targets = list(inst.targets_copy())
        decomp_inst = {"p": inst.args_copy()[0], "detectors": [], "observables": []}

        # separate detectors, logical observables, and separators
        for t in targets:
            if t.is_separator():
                continue
            elif t.is_logical_observable_id():
                # logical observable: L#
                decomp_inst["observables"].append(t.val)
            elif t.is_relative_detector_id():
                # detector: D#
                decomp_inst["detectors"].append(t.val)
        
        total_num_detectors = len(decomp_inst["detectors"])

        # iterate through array and make pairwise edge tuples with probability prob_err
        detectors = decomp_inst["detectors"]
        edges = []

        if total_num_detectors == 1:
            edges = [(-1, detectors[0])] # include a boundary edge
        
        else: # star decompose
            center_node = detectors[0]
            for i in range(total_num_detectors-1):
                edges.append(tuple(sorted([center_node, detectors[i+1]])))
        
        # store result
        decomp_inst["detectors"] = edges
        return decomp_inst

    # 
    # Edge decomposition tables 
    #

    def get_joint_prob(self, dem):
        """ Creates an array of joint probabilities representing edges in the DEM. Each entry [E][F] is the joint probability of edges E and detector F. 
            The diagonal entries [E][E] are the marginal probabilities of one graphlike error mechanism. The joint probabilities are calculated using the bernoulli formula for combining 
            probabilities when two detectors share more than one hyperedge.

            :param dem: stim.DetectorErrorModel object. The detector error model of the circuit to be used in decoding.
            :return: joint_probs: dictionary {[edge 1][edge 2]: joint probability} The joint probability matrix. Each cell is the joint probability of two detectors.
        """

        
        joint_probs = {} # each entry is the joint probability of two edges. [E][E] is a marginal probability
        fault_ids = {} # each entry is the fault id for that edge

        # iterate through each edge in the dem, add hyperedges
        for inst in dem:
            if inst.type == "error":
                decomposed_inst = self.decompose_dem_instruction_stim(inst) # used to be pairwise
                prob_err = decomposed_inst["p"]
                edges = decomposed_inst["detectors"]
                observables = decomposed_inst["observables"]

                # update hyperedges in joint probability table
                if len(edges) > 1:
                    a, b = edges[0], edges[1]
                    p01 = joint_probs.get(a, {}).get(b, 0)
                    p10 = joint_probs.get(b, {}).get(a, 0)

                    new_p01 = self.bernoulli_prob(p01, prob_err)
                    new_p10 = self.bernoulli_prob(p10, prob_err)

                    joint_probs.setdefault(a, {})[b] = new_p01
                    joint_probs.setdefault(b, {})[a] = new_p10

                # update marginal probabilities
                for i,edge in enumerate(edges):
                    p = joint_probs.get(edge, {}).get(edge, 0)
                    new_p = self.bernoulli_prob(p, prob_err)
                    joint_probs.setdefault(edge, {})[edge] = new_p
                    
                    # assign fault ids
                    obs = observables[i]
                    # obs = observables
                    fault_ids[edge] = fault_ids.get(edge) or obs
                
        return joint_probs, fault_ids 
    
    def get_conditional_prob(self, joint_prob_dict, decompose_biased):
        """ Given a joint probability dictionary, calculates the conditional probabilities for each hyperedge. The conditional probability is given by 
            P(A|B) = P(A^B)/P(A)
            Where A and B are edges from decomposed hyperedges. The marginal probability is P(A), and the joint probability is P(A^B). The maximum conditional probability is 0.5
            Only hyperedge components are present in final dictionary.

            :param joint_prob_dict: the joint probability of decomposed hyperedge between edges A and B
            :return: conditional probability nested dictionary. Of the same form as joint_prob_dict:
                    {edge tuple 1:{edge tuple 1: marginal probability, edge tuple two: conditional probability, P(edge 2 | edge 1), ...}, ...} 
        """

        cond_prob_dict = {}

        for edge_1 in joint_prob_dict:
            # find P(A)
            marginal_p = joint_prob_dict.get(edge_1, {}).get(edge_1,0)
            if marginal_p == 0:
                continue

            adjacent_edge_dict = joint_prob_dict.get(edge_1, {})

            # populate cond_prob dictionary 
            for edge_2 in adjacent_edge_dict: # in the other function, e1 is edge in correction and e2 is the edge affected. Here it is different

                if edge_1 == edge_2:  
                    continue 

                joint_p = joint_prob_dict.get(edge_1, {}).get(edge_2,0)
                edge_check_type = self.edge_type_d[edge_2] # have to make sure this is populated by the time I populate
                # print(edge_check_type, edge_2)

                scale = 1
                if decompose_biased:
                    if edge_check_type == "X": # edge_2 is a Z error since it's checks are X type.
                        scale = self.eta/(self.eta + 1)
                    elif edge_check_type == "Z": # edge_2 is an X error
                        scale = 1/2*(self.eta + 1)


                # conditional probability calculation. Min taken because weights cannot be negative, and eta=0.5 represents a full erasure channel
                # cond_p = min(1/(2*self.eta + 1), joint_p/marginal_p) # how do I do directionality here / I might have to think about it, will this actually work? Dont wanna fully erase edges...?
                cond_p = min(0.5, scale*joint_p/marginal_p) # trying to include the channel, not sure about directionaility still
                cond_prob_dict.setdefault(edge_1, {})[edge_2] = cond_p
        return cond_prob_dict

    #
    # Graph construction and DEM editing
    #

    def edit_dem(self, edges_in_correction, dem, cond_prob_dict):
        """ Given a stim DEM, updates the probabilities in error instructions with detectors given by cond_prob_dict based on detectors fired in correction.
            If a detector edge picked in the correction has a key in cond_prob_dict, it belonged to a hyperedge. The conditional probability then overwrites
            the original DEM probability for that hyperedge. Logical observables are distributed across new error instructions as in the original instruction.
        """
        # get a list of corrected edges from the first round
        edges_in_correction = [tuple(sorted(edge)) for edge in edges_in_correction]

        # iterate through the dem and fix the probabilities if they're in the cond_prob_dict
        # Create new DEM with updated probabilities
        new_dem = stim.DetectorErrorModel()

        for inst in dem:
            if inst.type == "error":
                old_prob = inst.args_copy()[0]
                decomposed_inst = self.decompose_dem_instruction_pairwise(inst)
                
                if len(decomposed_inst["detectors"]) > 1: # if the edge is a hyperedge
                    
                    for edge_1 in decomposed_inst["detectors"]: # break each hyperedge into sub-edges with their conditional prob
                        new_prob = old_prob
                        for edge_2 in edges_in_correction: # check which conditional probability is highest out of hyperedges
                            curr_prob = cond_prob_dict.get(edge_2, {}).get(edge_1,0)
                            new_prob = max(curr_prob, new_prob)
                        
                        targets = [stim.target_relative_detector_id(node) for node in edge_1] # will I have a problem with value -1?

                        if len(decomposed_inst["observables"]) > 0:
                            targets += [stim.target_logical_observable_id(l) for l in decomposed_inst["observables"]]
                        
                        new_inst = stim.DemInstruction("error", [new_prob], targets) # targets in edge_1 only
                        new_dem.append(new_inst)
                    
                    
                else: # if the edge is not a hyperedge, leave it be
                    new_dem.append(inst) 
            else:
                new_dem.append(inst)  # Preserve non-error instructions like detectors or shifts


        return new_dem

    def compute_edge_weights_from_conditional_probs(self, correction_edges, match_graph, cond_prob_dict, fault_ids_dict):
        weights = {}
        fault_ids = {}
        all_edges = match_graph.edges()
        # print(fault_ids_dict)
        edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        for u,v,data in all_edges:
            # print(u,v)
            e2 = tuple(sorted([-1 if x is None else x for x in (u, v)]))
            # print(e2)
            log_error = fault_ids_dict.get(e2, None)
            # print(log_error)
            p = max((cond_prob_dict.get(e1, {}).get(e2, 0) for e1 in edges_in_correction), default=0)
            if p > 0:
                weight = np.log((1-p)/p) 
                # print(f"updating edge {(u,v)} with conditional probability {p} and weight {weight}, from weight {data['weight']}")
            else:
                weight = data['weight']
            weights[(u, v)] = weight
            # fault_ids[(u, v)] = set([log_error]) if log_error is not None else set()
            fault_ids[(u, v)] = data['fault_ids']
            # this fault id has indices with (node, None): set(id)
        # print(fault_ids)
        return weights, fault_ids
    
    def compute_edge_weights_from_comp_gap(self, correction_edges, comp_correction_edges, matching, unsigned_gap, cutoff):
        """ Adjust the edge weights based on the complementary gap obtained during first pass matching.
            Use the signed gap to determine whether to use the min weight or complementary correction. 

            :param correction_edges(list): list of node pairs that represent the edges in the first MWPM pass
            :param comp_correction_edges(list): list of node pairs that represent the spatial complementary error in MWPM first pass
            :param matching(Matching): the matching graph to be updated
            :param unsigned_gap(float): magnitude represents first pass decoder confidence (sum of weights). 
            :param cutoff(float): the gap magnitude that is lower than the relative weights. Determines whether
                                we assign the gap to the complementary or minimum error path.
            :return: the weights and fault_ids dictionary recording the adjusted weight for each edge in the matchgraph
        """

        weights = {}
        fault_ids = {}

        sorted_edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        sorted_comp_correction_edges = [tuple(sorted(edge)) for edge in comp_correction_edges]

        mwpm_correction = [edge for edge in sorted_edges_in_correction if edge not in sorted_comp_correction_edges]
        comp_correction = [edge for edge in sorted_comp_correction_edges if edge not in sorted_edges_in_correction]

        edge_weight_dB_scale = self.get_dB_scaling(matching)
        
        for u,v,data in matching.edges():
            # fix the boundary nodes comparison because pymatching is inconsistent
            edge = tuple([u if (v is not None) else -1, v if (v is not None) else u])
            
            if np.abs(edge_weight_dB_scale*unsigned_gap) <= cutoff: # when the confidence is low choose the complementary path
                if edge in mwpm_correction:
                    weights[(u,v)] = 1e6
                else:
                    weights[(u,v)] = data['weight']
            else:
                weights[(u,v)] = data['weight'] # maybe try the other way later ... 
                # if edge in comp_correction:
                #     weights[(u,v)] = np.abs(edge_weight_dB_scale*signed_gap)
                # else:
                #     weights[(u,v)] = data['weight']
        
            fault_ids[(u,v)] = data['fault_ids']
        return weights, fault_ids
    
    def compute_edge_weights_all_correlated_info(self, correction_edges, matching, unsigned_gap, cond_prob_dict, fault_ids_dict):
        # definitely add stopping conditions if decoder is already right
        # when getting the hyperedge corrections, check if the comp decoder was right first too 
        
        weights = {}
        fault_ids = {}
        edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        for u,v,data in matching.edges():

            # edges in the correction get adjusted by unsigned gap
            if (u,v) in edges_in_correction:
                print(f"{u,v} weight adjusted by unsigned gap")
                weight = 1/unsigned_gap

            # edges not in the correction get hyperedge adjustments
            else:
                e2 = tuple(sorted([-1 if x is None else x for x in (u, v)])) # get (u,v) to the proper bndry format given my code
                
                # find the max conditional probability adjustment for this edge given the correction
                p = max((cond_prob_dict.get(e1, {}).get(e2, 0) for e1 in edges_in_correction), default=0) 

                if p > 0:
                    print(f"{u,v} weight adjusted by cond prob")
                    weight = np.log((1-p)/p) 
                else:
                    weight = data['weight']
            fault_ids[(u, v)] = data['fault_ids']
            weights[(u, v)] = weight

        return weights, fault_ids


    def build_matching_from_weights(self, weights_dict, fault_ids_dict, original_num_nodes):
        match = Matching()
        for (u, v), weight in weights_dict.items():
            # fault_id = fault_ids_dict.get(tuple([u if v is not None else -1, v if v is not None else u]), None)
            # print(fault_id, u,v)
            fault_id = fault_ids_dict.get((u,v),None)
            if None in (u, v):
                match.add_boundary_edge(u if u is not None else v, weight=weight, fault_ids=fault_id)
            else:
                match.add_edge(u, v, weight=weight, fault_ids=fault_id)
        
        # Now detect which nodes were never added via any edge
        used_nodes = set()
        for (u, v) in weights_dict.keys():
            if u is not None:
                used_nodes.add(u)
            if v is not None:
                used_nodes.add(v)

        # Fill in unused detector nodes (not involved in any edge)
        all_nodes = set(range(original_num_nodes))
        missing_nodes = all_nodes - used_nodes

        for node in missing_nodes:
            # Use an extremely high weight to ensure these edges are not used
            match.add_boundary_edge(node, weight=1e6)
        
        return match


    #
    # Decoding
    #

    def decoding_failures_correlated_circuit_level(self, circuit, shots, mem_type, CD_type, decompose_biased=True):
        """
        Finds the number of logical errors given a circuit using correlated decoding. Uses pymatching's correlated decoding approach, inspired by
        papers cited in the README.
        :param circuit: stim.Circuit object, the circuit to decode
        :param p: physical error rate
        :param shots: number of shots to sample
        :param memtype: basis to run memory experiment
        :param CD_type: the clifford deformation type applied to the code
        :param decompose_biased: whether to decompose hyperedges with bias in mind or give equal weight to X and Z components
        :return: number of logical errors
        """

        # 
        # Get the edge data for correlated decoding
        #

        # get the DEM get the matching graph
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True, approximate_disjoint_errors=True)
        matchgraph = Matching.from_detector_error_model(dem, enable_correlations=False)
        self.edge_type_d = self.get_edge_type_d(dem, mem_type, CD_type)

        # get the joint probabilities table of the dem hyperedges
        joint_prob_dict, fault_ids = self.get_joint_prob(dem)
        
        # calculate the conditional probabilities based on joint probablities and marginal probabilities 
        cond_prob_dict = self.get_conditional_prob(joint_prob_dict, decompose_biased)

        # instead of performing the first round of error correction and going based on this, create a MWPM graph based on hyperedges in joint_prob_dict

        # new_dem = edit_dem() 

        
        
        #
        # Decode the circuit
        #
        
        # first round of decoding
        # get the syndromes and observable flips
        seed = np.random.randint(0, 2**32 - 1)
        sampler = circuit.compile_detector_sampler(seed=seed)
        syndrome, observable_flips = sampler.sample(shots, separate_observables=True)
        # print("syndrome inside function:", syndrome )

        # from eva
        # change the logicals so that there is an observable for each qubit, change back to the code cap case to check whether the real logical flipped

        corrections = np.zeros((shots, 2)) # largest fault id is 1, len of correction = 2
        for i in range(shots):

            # print(syndrome[i].shape)
            edges_in_correction = matchgraph.decode_to_edges_array(syndrome[i])
            # print("edges in correction inside function from mycorr", edges_in_correction)

            
            # update weights based on conditional probabilities
            # updated_dem = self.edit_dem(edges_in_correction, dem, cond_prob_dict) # is this DEM updated correctly? make sure that it is getting the right edges

            # second round of decoding with updated weights
            # matching_corr = Matching.from_detector_error_model(updated_dem, enable_correlations=False)
            updated_weights, fault_ids_dict = self.compute_edge_weights_from_conditional_probs(edges_in_correction, matchgraph, cond_prob_dict, fault_ids)
            matching_corr = self.build_matching_from_weights(updated_weights, fault_ids_dict, matchgraph.num_nodes)
            # print("updated edges inside function from mycorr", matching_corr.edges())
            # print(matching_corr.decode(syndrome[i]).shape, matching_corr.decode(syndrome[i]))
            corrections[i] = matching_corr.decode(syndrome[i]) #usual code

        
        # calculate the number of logical errors
        log_errors_array = np.any(np.array(observable_flips) != np.array(corrections), axis=1) # usual code
        return log_errors_array


    def decoding_failures_correlated_gap(self, circuit, shots, mem_type, CD_type, cutoff=1):
        """
        Two stage decoding following arxiv:2312.04522., with the addition of a hyperedge decoding step.
        """

        # get the hyperedge data + set up original matching
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True, approximate_disjoint_errors=True)
        matchgraph = Matching.from_detector_error_model(dem, enable_correlations=False)
        self.edge_type_d = self.get_edge_type_d(dem, mem_type, CD_type)

        # get the joint probabilities table of the dem hyperedges
        joint_prob_dict, fault_ids = self.get_joint_prob(dem)
        
        # calculate the conditional probabilities based on joint probablities and marginal probabilities 
        cond_prob_dict = self.get_conditional_prob(joint_prob_dict, decompose_biased=False)



        #
        # Decode the circuit
        #
        
        # first round of decoding
        # get the syndromes and observable flips
        seed = np.random.randint(0, 2**32 - 1)
        sampler = circuit.compile_detector_sampler(seed=seed) # should I be passing in a seed instead so I am comparing LER of right shots?
        detection_events, observable_flips = sampler.sample(shots, separate_observables=True)

        
        corrections = np.zeros(observable_flips.shape)
        for shot in range(shots):
            us_gap, edges_in_correction, edges_in_comp_correction, pred_min, pred_picked = self.get_complementary_correction(dem, detection_events[shot], observable_flips[shot], return_predictions=True)

            
            # when the first pass of MWPM is not confident, get the complementary graph 
            if us_gap < cutoff:
                comp_weights,comp_fault_ids = self.compute_edge_weights_from_comp_gap(edges_in_correction,edges_in_comp_correction, matchgraph, us_gap, cutoff)
                comp_matching = self.build_matching_from_weights(comp_weights, comp_fault_ids, matchgraph.num_nodes)

                # hyperedge adjustment based on comp correction
                hyperedge_weights, hyperedge_fault_ids = self.compute_edge_weights_from_conditional_probs(edges_in_comp_correction,
                                                                                                                comp_matching,
                                                                                                                cond_prob_dict,
                                                                                                                comp_fault_ids)

            else: # the first correction is confident, just do regular hyperedge decomposition on correction
                hyperedge_weights, hyperedge_fault_ids = self.compute_edge_weights_from_conditional_probs(edges_in_correction,
                                                                                                                matchgraph,
                                                                                                                cond_prob_dict,
                                                                                                                fault_ids)
            hyperedge_matching = self.build_matching_from_weights(hyperedge_weights, hyperedge_fault_ids,matchgraph.num_nodes)
            
            corrections[shot] = hyperedge_matching.decode(detection_events[shot])
        
        log_errors_array = np.any(np.array(observable_flips) != np.array(corrections), axis=1)

        return log_errors_array


    #
    #
    # Circuit sampling functions
    #
    #

    def get_num_log_errors(self, circuit, num_shots):
        """
        Get the number of logical errors from a circuit phenom. model, not the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :return: logical errors array. Sum of array is the number of logical errors
        """
        matching = Matching.from_stim_circuit(circuit)
        seed = np.random.randint(0, 2**32 - 1)
        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matching.decode_batch(detection_events)
        
        
        num_errors_array = np.zeros(num_shots)
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors_array[shot] = 1
        return num_errors_array

    def get_num_log_errors_DEM(self, circuit, num_shots, enable_corr, enable_pymatch_corr, meas_type, CD_type="SC"):
        """
        Get the number of logical errors from the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :param enable_corr: boolean whether to use house-made correlated decoder
        :param enable_pymatch_corr: boolean whether to use pymatching correlated decoder
        :return: number of logical errors
        """
        if enable_corr:
            # house-made circuit level correlated decoder
            log_errors_array = self.decoding_failures_correlated_circuit_level(circuit, num_shots, meas_type, CD_type)

        
        else: # no correlated decoding or pymatching correlated decoding
            dem = circuit.detector_error_model(decompose_errors=enable_pymatch_corr, approximate_disjoint_errors=True)
            matchgraph = Matching.from_detector_error_model(dem,enable_correlations=enable_pymatch_corr)
            seed = np.random.randint(0, 2**32 - 1)
            sampler = circuit.compile_detector_sampler(seed=seed) # double check that this randomness is doing the right thing, every shot should be random and compare
            syndrome, observable_flips = sampler.sample(num_shots, separate_observables=True) # do i need to set a seed here?
            predictions = matchgraph.decode_batch(syndrome, enable_correlations=enable_pymatch_corr) # had a weird recent error, should have thrown an error earlier when I passed in enable correlations
            log_errors_array = np.any(np.array(observable_flips) != np.array(predictions), axis=1)
        
        return log_errors_array

    def get_log_error_circuit_level(self, p_list, meas_type, num_shots, noise_model="code_cap", cd_type="SC", corr_decoding= False, pymatch_corr = False):
        """
        Get the logical error rate for a list of physical error rates of gates at the circuit level
        :param p_list: list of p values
        :param meas_type: type of stabilizers measured in memory experiment. Meas type X indicates ZL detection for Z errors
        :param num_shots: number of shots to sample
        :param noise_model: the noise model to use, either "code_cap", "phenom", or "circuit_level". Code cap has a biased depolarizing channel on data 
            qubits at the beginning of rounds. Phenominological model has a biased depolarizing channel on data qubits at the beginning of rounds and bit-flip noise on 
            measurement qubits before measurement. Circuit level has biased depolarizing channel at the beginning of rounds, bit-flip noise on measurement qubits before measurement, 
            and a two-qubit depolarizing channel after each two-qubit clifford gate.
        :param cd_type: the type of clifford defomation applied to the circuit. Either None, XZZXonSqu, or ZXXZonSqu.
        :return: list of logical error rates, opposite type of the measurement type (e.g. if meas_type is X, then Z logical errors are returned)
        """
        

        log_error_L = []
        for p in p_list:
            # make the circuit
            circuit_obj = cc_circuit.CDCompassCodeCircuit(self.d, self.l, self.eta, meas_type) # change list of ps dependent on model
            if noise_model == "code_cap":# change this based on the noise model you want
                circuit = circuit_obj.make_elongated_circuit_from_parity(0,0,0,p,0,0,CD_type=cd_type, memory=False)  
            elif noise_model == "phenom":
                circuit = circuit_obj.make_elongated_circuit_from_parity(p,0,0,p,0,0,CD_type=cd_type, phenom_meas=True) # check the plots that matched pymatching to get error model right, before meas flip and data qubit pauli between rounds
            elif noise_model == "circuit_level":
                circuit = circuit_obj.make_elongated_circuit_from_parity(before_measure_flip=p,before_measure_pauli_channel=0,after_clifford_depolarization=p,before_round_data_pauli_channel=0,between_round_idling_pauli_channel=p,idling_dephasing=0,CD_type=cd_type) # between round idling biased pauli on all qubits, measurement flip errors, 2-qubit gate depolarizing
            else:
                raise ValueError("Invalid noise model. Choose either 'code_cap', 'phenom', or 'circuit_level'.")
            
            log_errors_array = self.get_num_log_errors_DEM(circuit, num_shots, corr_decoding, pymatch_corr, meas_type, cd_type)
            log_error_L.append(log_errors_array)

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

# def get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data, noise_model="circuit_level", cd_type="SC", corr_decoding=False, pymatch_corr=False):
#     """ Generate logical error rates for x,z, correlatex z, and total errors
#         via MC sim in decoding_failures_correlated and add it to a shared pandas df
        
#         in: num_shots - the number of MC iterations
#             l - the integer repition of the compass code
#             eta - the float bias ratio of the error model
#             p_list - array of probabilities to scan
#             d_list - the distances of compass code to scan
        
#         out: a pandas df recording the logical error rate with all corresponding params

#     """
#     # print(f"in get data,  l = {l}, eta = {eta}, corr_type = {corr_type}, num_shots = {num_shots}, noise_model = {noise_model}, cd_type = {cd_type}")
#     err_type = {0:"X", 1:"Z", 2:corr_type, 3:"TOTAL"}
#     if circuit_data:
#         data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "noise_model": [], "CD_type":[], "num_log_errors":[], "time_stamp":[]}
#     else:
#         data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "num_log_errors":[], "time_stamp":[]}
#     data = pd.DataFrame(data_dict)

#     for d in d_list:
#         if circuit_data:
#             # print("running circuit data")
            
#                 # circuit_x = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "X")
#                 # circuit_z = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "Z")
    
#             decoder = CorrelatedDecoder(eta, d, l, corr_type)
#             log_errors_z_array = decoder.get_log_error_circuit_level(p_list, "Z", num_shots, noise_model, cd_type, corr_decoding, pymatch_corr) # get the Z logical errors from Z memory experiment, X errors
#             log_errors_x_array = decoder.get_log_error_circuit_level(p_list, "X", num_shots, noise_model, cd_type, corr_decoding, pymatch_corr) # get the X logical errors from X memory experiment, Z errors
#             log_errors_z = np.sum(log_errors_z_array, axis=1) # double counting fs fs
#             log_errors_x = np.sum(log_errors_x_array, axis=1)        
#             log_errors_total = np.sum(np.logical_or(log_errors_x_array, log_errors_z_array), axis=1)



#             for i,log_error in enumerate(log_errors_x):
#                 if pymatch_corr:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 elif corr_decoding:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 else:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                
#                 data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)

#             for i,log_error in enumerate(log_errors_z):
#                 if pymatch_corr:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 elif corr_decoding:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 else:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)

#             for i,log_error in enumerate(log_errors_total):
#                 if pymatch_corr:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 elif corr_decoding:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
#                 else:
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                
#                 data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
            
            

#         else:
#             decoder = CorrelatedDecoder(eta, d, l, corr_type)

#             for p in p_list:
#                 errors = decoder.decoding_failures_correlated(p, num_shots)
#                 for i in range(len(errors)):
#                     curr_row = {"d":d, "num_shots":num_shots, "p":p, "l": l, "eta":eta, "error_type":err_type[i], "num_log_errors":errors[i]/num_shots, "time_stamp":datetime.now()}
#                     data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
#     return data


# def shots_averaging(num_shots, l, eta, err_type, in_df, CD_type, file):
#     """For the inputted number of shots, averages those shots over the array length run on computing cluster.  
#         in: num_shots - int, the number of monte carlo shots in the original simulation
#             arr_len -  int, the number of jobs / averaging interval desired
#             l - int, elongation parameter
#             eta - float, noise bias
#             err_type - the type of error to average
#             df - the dataframe of interest. If None, read from the CSV file
#     """
#     if in_df is None:
#         in_data = pd.read_csv(file)
#         data = in_data[(in_data['num_shots'] == num_shots) & (in_data['l'] == l) &(in_data['eta'] == eta) & (in_data['error_type'] == err_type) & (in_data['CD_type'] == CD_type)]
#     else:
#         data = in_df
#     data_mean = data.groupby('p', as_index=False)['num_log_errors'].mean()
#     return data_mean



# def write_data(num_shots, d_list, l, p_list, eta, ID, corr_type, circuit_data, noise_model="code_cap", cd_type="SC", corr_decoding=False, pymatch_corr=False):
#     """ Writes data from pandas df to a csv file, for use with SLURM arrays. Generates data for each slurm output on a CSV
#         in: num_shots - the number of MC iterations
#             l - the integer repition of the compass code
#             eta - the float bias ratio of the error model
#             p_list - array of probabilities to scan
#             d_list - the distances of compass code to scan
#             ID - SLURM input task_ID number, corresponds to which array element we run
#     """
#     # print(f"in write data, ID = {ID}, l = {l}, eta = {eta}, corr_type = {corr_type}, num_shots = {num_shots}, noise_model = {noise_model}, cd_type = {cd_type}")
#     data = get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data, noise_model=noise_model, cd_type=cd_type, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)
#     if circuit_data:
#         if pymatch_corr:
#             data_file = f'circuit_data/py_corr_{ID}.csv'
#             if not os.path.exists('circuit_data/'):
#                 os.mkdir('circuit_data')
#         else:
#             data_file = f'circuit_data/circuit_level_{ID}.csv'
#             if not os.path.exists('circuit_data/'):
#                 os.mkdir('circuit_data')
#     else:
#         data_file = f'corr_err_data/code_cap_{ID}.csv'
#         if not os.path.exists('corr_err_data/'):
#             os.mkdir('corr_err_data')
   
    

#     # Check if the CSV file exists
#     if os.path.isfile(data_file):
#         # If it exists, load the existing data
#         past_data = pd.read_csv(data_file)
#         # Append the new data
#         all_data = pd.concat([past_data, data], ignore_index=True)
#     else:
#         # If it doesn't exist, the new data will be the combined data
#         all_data = data
#     # Save the combined data to the CSV file
#     all_data.to_csv(data_file, index=False)

def get_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    corr_type,
    circuit_data,
    noise_model="circuit_level",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    data_file=None,
    append=False,
    chunk_size=5000,
):
    """Generate logical error-rate data in chunks.

    For each (d, p), run in chunks of size `chunk_size`, append each chunk's
    result to CSV immediately, and return a dataframe of all rows generated.

    Note:
        The `num_log_errors` column is preserved as the logical error RATE
        within each chunk, matching your existing files.
    """
    err_type = {0: "X", 1: "Z", 2: corr_type, 3: "TOTAL"}

    if circuit_data:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "noise_model", "CD_type", "num_log_errors", "time_stamp"
        ]
    else:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "num_log_errors", "time_stamp"
        ]

    all_rows = []

    def flush_rows(rows_to_write):
        """Append rows to CSV immediately and force flush to disk."""
        if not rows_to_write:
            return

        if append and data_file is not None:
            chunk_df = pd.DataFrame(rows_to_write, columns=columns)
            file_exists = os.path.isfile(data_file)
            chunk_df.to_csv(
                data_file,
                mode="a",
                header=not file_exists,
                index=False,
            )

            # Force the OS buffer to flush as much as possible
            with open(data_file, "a") as f:
                f.flush()
                os.fsync(f.fileno())

    for d in d_list:
        decoder = CorrelatedDecoder(eta, d, l, corr_type)

        for p in p_list:
            shots_done = 0

            while shots_done < total_num_shots:
                curr_num_shots = min(chunk_size, total_num_shots - shots_done)

                print(
                    f"Running d={d}, p={p}, eta={eta}, l={l}, "
                    f"shots {shots_done} -> {shots_done + curr_num_shots}"
                )

                if circuit_data:
                    # Run one p at a time, one chunk at a time
                    log_errors_z_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "Z",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                    )
                    log_errors_x_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "X",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                    )

                    log_errors_z = np.sum(log_errors_z_array, axis=1)[0]
                    log_errors_x = np.sum(log_errors_x_array, axis=1)[0]
                    log_errors_total = np.sum(
                        np.logical_or(log_errors_x_array, log_errors_z_array),
                        axis=1,
                    )[0]

                    if pymatch_corr:
                        x_err_type = "X_MEM_PY"
                        z_err_type = "Z_MEM_PY"
                        total_err_type = "TOTAL_MEM_PY"
                    elif corr_decoding:
                        x_err_type = "X_MEM_CORR"
                        z_err_type = "Z_MEM_CORR"
                        total_err_type = "TOTAL_MEM_CORR"
                    else:
                        x_err_type = "X_MEM"
                        z_err_type = "Z_MEM"
                        total_err_type = "TOTAL_MEM"

                    rows_for_chunk = [
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": x_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_x / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": z_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_z / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": total_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_total / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                    ]

                else:
                    # Code-capacity: one p at a time, one chunk at a time
                    errors = decoder.decoding_failures_correlated(p, curr_num_shots)

                    rows_for_chunk = []
                    for i in range(len(errors)):
                        rows_for_chunk.append({
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": err_type[i],
                            "num_log_errors": errors[i] / curr_num_shots,
                            "time_stamp": datetime.now(),
                        })

                all_rows.extend(rows_for_chunk)
                flush_rows(rows_for_chunk)

                shots_done += curr_num_shots

                print(
                    f"Saved d={d}, p={p}, eta={eta}, l={l}, "
                    f"chunk_shots={curr_num_shots}, total_done={shots_done}/{total_num_shots}"
                )

    return pd.DataFrame(all_rows, columns=columns)


def write_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    ID,
    corr_type,
    circuit_data,
    noise_model="code_cap",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    chunk_size=500,
    overwrite=True,
):
    """Write data incrementally to CSV while the job runs.

    Parameters
    ----------
    total_num_shots : int
        Total number of shots desired for each (d, p).
    chunk_size : int
        Number of shots to run before checkpointing to CSV.
    overwrite : bool
        If True, delete an existing file with the same ID before starting.
    """
    if circuit_data:
        os.makedirs("circuit_data", exist_ok=True)
        if pymatch_corr:
            data_file = f"circuit_data/py_corr_{ID}.csv"
        else:
            data_file = f"circuit_data/circuit_level_{ID}.csv"
    else:
        os.makedirs("corr_err_data", exist_ok=True)
        data_file = f"corr_err_data/code_cap_{ID}.csv"

    if overwrite and os.path.isfile(data_file):
        os.remove(data_file)

    data = get_data(
        total_num_shots=total_num_shots,
        d_list=d_list,
        l=l,
        p_list=p_list,
        eta=eta,
        corr_type=corr_type,
        circuit_data=circuit_data,
        noise_model=noise_model,
        cd_type=cd_type,
        corr_decoding=corr_decoding,
        pymatch_corr=pymatch_corr,
        data_file=data_file,
        append=True,
        chunk_size=chunk_size,
    )

    return data


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
    output_file_CL = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data.csv'
    
    all_data_XZ = pd.DataFrame()
    all_data_ZX = pd.DataFrame()
    all_data_CL = pd.DataFrame()

    xz_exists = os.path.exists(output_file_XZ)
    zx_exists = os.path.exists(output_file_ZX)
    cl_exists = os.path.exists(output_file_CL)

    # Check if the output file already exists
    if not circuit_data:
        if xz_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_XZ)
            # Append the new data to the existing data
            all_data_XZ = pd.concat([existing_data, new_data_XZ], ignore_index=True)
        elif not xz_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_XZ = new_data_XZ

        if zx_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_ZX)
            # Append the new data to the existing data
            all_data_ZX = pd.concat([existing_data, new_data_ZX], ignore_index=True)
        elif not zx_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_ZX = new_data_ZX
        all_data_XZ.to_csv(output_file_XZ, index=False)
        all_data_ZX.to_csv(output_file_ZX, index=False)

    else:
        if cl_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_CL)
            # Append the new data to the existing data
            all_data_CL = pd.concat([existing_data, new_data_CL], ignore_index=True)
        else:
            # If the file doesn't exist, the new data is the combined data
            all_data_CL = output_file_CL
        
        all_data_CL.to_csv(output_file_CL, index=False)

    
    for file in data_files:
        os.remove(file)

def full_error_plot(full_df, curr_eta, curr_l, curr_num_shots, noise_model, CD_type, file, corr_decoding=False, py_corr=False, loglog=False, averaging=True, circuit_level=False, plot_by_l=False):
    """Make a plot of all errors given a df with unedited contents of an entire CSV.
        :param full_df: pandas DataFrame with unedited contents from CSV
        :param curr_eta: current noise bias to filter DataFrame
        :param curr_l: current elongation parameter to filter DataFrame
        :param curr_num_shots: current number of shots to filter DataFrame
        :param noise_model: the type of simulation, either "code_cap", "phenom", or "circuit_level"
        :param CD_type: the type of clifford deformation used, from a list ["SC", "XZZXonSqu", "ZXXZonSqu"]
        :param py_corr: boolean whether pymatching correlated decoding was used, chooses from last of list ["CORR_XZ", "CORR_ZX", "X_MEM", "Z_MEM", "TOTAL_MEM", "X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"]
        :param file: the CSV file path, used for averaging shots if in_df is None
        :param loglog: boolean whether to use loglog scale for plotting
        :param averaging: boolean whether to average shots over the number of jobs
        :param circuit_level: boolean whether the data is from circuit level simulations. Alternative is vector simulation.
        :param plot_by_l: boolean whether to plot by elongation parameter l instead of error type

        :return: no return, shows a matplotlib plot
    """

    # prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    # filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)] 
                    # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
    
    filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots) & (full_df['noise_model'] == noise_model) & (full_df['CD_type'] == CD_type)]

    if py_corr: 
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_PY', 'Z_MEM_PY', 'TOTAL_MEM_PY'])]
    elif corr_decoding:
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR'])]
    else:
        if circuit_level:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM', 'Z_MEM', 'TOTAL_MEM'])]
        else:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X', 'Z', 'TOTAL', 'CORR_XZ', 'CORR_ZX'])]

    # Get unique error types and unique d values
    error_types = filtered_df['error_type'].unique()

    d_values = filtered_df['d'].unique()


    # Create a figure with subplots for each error type
    if len(error_types)%2 == 0:
        fig, axes = plt.subplots(len(error_types)//2, 2, figsize=(15, 5*len(error_types)//2))
    else:
        fig, axes = plt.subplots((len(error_types)//2)+1, 2, figsize=(15, 5*((len(error_types)//2)+1)))
    axes = axes.flatten()
    

    # Plot each error type in a separate subplot
    for i, error_type in enumerate(error_types):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
        ax.tick_params(axis='both', which='minor', labelsize=16)  
        error_type_df = filtered_df[filtered_df['error_type'] == error_type]
        prob_scale = get_prob_scale(error_type, curr_eta)
        # Plot each d value
        for d in d_values:
            d_df = error_type_df[error_type_df['d'] == d]
            if averaging:
                # to check that this is working, figure out how big this DF is
                d_df_mean = shots_averaging(curr_num_shots, curr_l, curr_eta, error_type, d_df, CD_type, file)
                if loglog:
                    ax.loglog(d_df_mean['p']*prob_scale[error_type], d_df_mean['num_log_errors'],  label=f'd={d}')
                    error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors']))
                    ax.fill_between(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2)
                else:
                    ax.plot(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}')
            else:
                ax.scatter(d_df['p']*prob_scale, d_df['num_log_errors'], s=2, label=f'd={d}')

        
        ax.set_title(f'Error Type: {error_type}', fontsize=20)
        ax.set_xlabel('p', fontsize=14)
        ax.set_ylabel('num_log_errors', fontsize=20)
        ax.legend()

    if circuit_level:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta}, l = {curr_l}, Deformation = {CD_type}')
    else:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()

def threshold_plot(full_df, p_th0, p_range, curr_eta, curr_l, curr_num_shots, corr_type, CD_type, noise_model, file, circuit_level=False, py_corr = False, corr_decoding=False, loglog=False, averaging=True, show_threshold=True, show_fit=False):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['p'] > p_th0 - p_range)&(full_df['p'] < p_th0 + p_range)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots) & (full_df['noise_model'] == noise_model) & (full_df['CD_type'] == CD_type)]
    
    if py_corr: 
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_PY', 'Z_MEM_PY', 'TOTAL_MEM_PY'])]
    elif corr_decoding:
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR'])]
    else:
        if circuit_level:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM', 'Z_MEM', 'TOTAL_MEM'])]
        else:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X', 'Z', 'TOTAL', 'CORR_XZ', 'CORR_ZX'])]
    
    
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
            d_df_mean = shots_averaging(curr_num_shots, curr_l, curr_eta, corr_type, d_df, CD_type, file)
            if loglog:
                ax.loglog(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
                error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors']))
                ax.fill_between(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2, color=colors[i])
            else:
                ax.plot(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
        else:
            ax.scatter(d_df['p']*prob_scale, d_df['num_log_errors'], s=2, label=f'd={d}',color=colors[i])

    popt, pcov = get_threshold(filtered_df, p_th0, p_range, curr_l, curr_eta, corr_type,curr_num_shots, CD_type)
    pth = popt[0]
    pth_error = np.sqrt(np.diag(pcov))[0]
    
    if show_threshold:
        ax.vlines(pth, ymin=0, ymax=max(filtered_df['num_log_errors']), color='red', linestyles='--', label=f'pth = {pth:.3f} +/- {pth_error:.3f}')
    if show_fit:
        for d in d_values:
            y_fit = []
            p_list = []
            for p in sorted(filtered_df['p'].unique()):
                x = (p, d)
                y_fit += [threshold_fit(x, *popt)]
                p_list += [p]
            if loglog:
                ax.loglog(np.array(p_list)*prob_scale, y_fit, linestyle='--', color='red')
            else:
                # print(p_list, y_fit)
                ax.plot(np.array(p_list)*prob_scale, y_fit, linestyle='--', color='red')

    
    ax.set_title(f'Error Type: {corr_type}', fontsize=20)
    ax.set_xlabel('p', fontsize=14)
    ax.set_ylabel('num_log_errors', fontsize=20)
    ax.legend()

    fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()


# def eta_threshold_plot(eta_df, cd_type, corr_type_list, noise_model):
#     """Make a single figure with a 2-column grid of subplots.
#     Each row corresponds to a different `l`, with CORR_XZ on left and CORR_ZX on right.
#     """
#     # print(eta_df)
#     # print("Unique cd_type in df:", eta_df['cd_type'].unique())
#     # print("cd_type being filtered for:", repr(cd_type))
#     # print("Unique noise_model in df:", eta_df['noise_model'].unique())
#     # print("noise_model being filtered for:", repr(noise_model))
#     eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
#     eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()

#     cd_type = cd_type.strip()
#     noise_model = noise_model.strip()
#     # print(cd_type, noise_model)
#     df = eta_df[(eta_df['CD_type'] == cd_type) &
#                 (eta_df['noise_model'] == noise_model)]
#     # print(df)
#     l_values = sorted(df['l'].unique())
#     num_rows = len(l_values)
#     num_cols = len(corr_type_list)

#     # Set up colors
#     cmap = colormaps['Blues_r']
#     color_values = np.linspace(0.1, 0.8, num_rows)
#     l_colors = [cmap(val) for val in color_values]

#     # Create figure and 2-column grid
#     # Create figure and subplot grid
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 2.5 * num_rows), sharex=True, sharey=True)

#     # Make axes 2D for consistent indexing
#     if num_rows == 1 and num_cols == 1:
#         axes = np.array([[axes]])
#     elif num_rows == 1:
#         axes = axes[np.newaxis, :]
#     elif num_cols == 1:
#         axes = axes[:, np.newaxis]

#     for row_idx, l in enumerate(l_values):
#         for col_idx, error_type in enumerate(corr_type_list):
#             ax = axes[row_idx, col_idx]
#             mask = (
#                 (df['l'] == l) &
#                 (df['error_type'] == error_type)
#             )
#             df_filtered = df[mask].sort_values(by='eta')

#             eta_vals = df_filtered['eta'].to_numpy()
#             pth_list = df_filtered['pth'].to_numpy()
#             pth_error_list = df_filtered['stderr'].to_numpy()

#             ax.errorbar(
#                 eta_vals, pth_list, yerr=pth_error_list,
#                 label=f'l = {l}', color=l_colors[row_idx],
#                 marker='o', capsize=5
#             )

#             if row_idx == 0:
#                 ax.set_title(f"{error_type}, Deformation {cd_type}", fontsize=16)

#             if col_idx == 0:
#                 ax.set_ylabel(f"l = {l}\nThreshold $p_{{th}}$", fontsize=12)

#             if row_idx == num_rows - 1:
#                 ax.set_xlabel("Noise Bias (η)", fontsize=12)

#             ax.grid(True)
#             ax.legend()

#     plt.tight_layout()
#     plt.show()

def eta_threshold_plot(eta_df, cd_type, corr_type_list, noise_model):
    """One subplot per corr_type, all l values overlaid, shaded error bands,
    single shared legend, deformation in title."""
    
    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()

    df = eta_df[
        (eta_df['CD_type'] == cd_type) &
        (eta_df['noise_model'] == noise_model)
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(corr_type_list)

    # Colors
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, len(l_values))
    l_colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(8.5 * num_cols, 4.8),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    # Store handles for shared legend
    legend_handles = []
    legend_labels = []

    

    for col_idx, error_type in enumerate(corr_type_list):
        ax = axes[col_idx]

        for l_idx, l in enumerate(l_values):
            mask = (
                (df['l'] == l) &
                (df['error_type'] == error_type)
            )
            df_filtered = df[mask].sort_values(by='eta')
            
            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            color = l_colors[l_idx]

            # Plot line
            line, = ax.plot(
                eta_vals,
                pth,
                label=f'l = {l}',
                color=color,
                marker='o'
            )

            # Shaded error
            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=color,
                alpha=0.2
            )

            # Only collect legend entries once
            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(f'l = {l}')


        parts = error_type.split("_")
        if len(parts) >= 2:
            title = rf"$\mathrm{{{parts[0]}}}_{{{parts[1]}}}$"
        else:
            title = rf"$\mathrm{{{error_type}}}$"

        ax.set_title(title, fontsize=16)
        # ax.set_title(f"{error_type}", fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

        # Global title
    fig.suptitle(
        f"Threshold vs Bias Pymatching Correlated Decoder (Deformation: {cd_type})",
        fontsize=18,
        y=0.98
    )

    # Shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    # Manually leave vertical space for suptitle + legend
    fig.subplots_adjust(top=0.78, wspace=0.12)

    plt.show()

# def threshold_fit(x, pth, nu, a, b, c):
#     p,d = x
#     X = (d**(1/nu))*(p-pth)
#     return c + b*X + a*X**2

def threshold_fit(x, pth, nu, a,b,c):
    p,d = x
    X = (d**(1/nu))*(p-pth)
    return a + b*X + c*X**2

def get_threshold(full_df, pth0, p_range, l, eta, error_type, num_shots, CD_type):
    """ returns the threshold and confidence given a df 
        in: df - the dataframe containing all data, filtered for one error_type, l eta, and probability range
        out: p_thr - a float, the probability where intersection of different lattice distances occurred
    """
    print(f"Getting threshold for l = {l}, eta = {eta}, error type = {error_type}, num_shots = {num_shots}, CD = {CD_type}")
    df = full_df[(full_df['p'] < pth0 + p_range) & ( full_df['p'] > pth0 - p_range) & (full_df['l'] == l) & (full_df['eta'] == eta) & (full_df['error_type'] == error_type) & (full_df['num_shots'] == num_shots) & (full_df['CD_type'] == CD_type)]
    # print(df.head)
    # df = full_df
    if df.empty:
        return 0, 0

    # get the p_list and d_list from the dataframe
    p_list = df['p'].to_numpy().flatten()
    d_list = df['d'].to_numpy().flatten()
    error_list = df['num_log_errors'].to_numpy().flatten()

    # run the fitting function
    # popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 0.5, 1, 1, 1])
    popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 0.5, 0,0,0])
    
    # pth = popt[0] # the threshold probability
    # pth_error = np.sqrt(np.trace(pcov))
    # overfitting = np.linalg.cond(pcov)
    # print(f"Overfitting condition number: {overfitting}")
    # print(f"diag of covariance matrix: {np.diag(pcov)}")
    return popt, pcov


def get_prob_scale(corr_type, eta):
    """ extract the amount to be scaled by given a noise bias and the type of error
    """
    prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'CORR_ZX':1, 'TOTAL':1, 'TOTAL_MEM':1, 'X_MEM':  1, 'Z_MEM': 1, 'TOTAL_MEM_PY':1, 'X_MEM_PY':1, 'Z_MEM_PY':1,'TOTAL_MEM_CORR':1, 'X_MEM_CORR':1, 'Z_MEM_CORR':1} # TOTAL_MEM 4/3 factor of total mem is due to code_cap pauli channel scalling factor in stim, remove this?
    return prob_scale[corr_type]


def get_data_DCC(circuit_data, corr_decoding, noise_model, d_list, l_list, eta_list, cd_list, corr_list, total_num_shots, p_list=None, p_th_init_d=None, pymatch_corr=False):
    """ Function to get the data from the DCC using parallel SLURM arrays. Each array task will get data for a specific (l, eta, corr_type) or (l, eta, cd_type) combo.
        The total number of shots will be split evenly across the array tasks so that the total number of shots is reached upon averaging. 
        in: circuit_data - boolean, whether to get data from circuit or vector code cap
            corr_decoding - boolean, whether to get data from correlated decoding or not
            noise_model - string, the noise model to use for circuit data, either "code_cap", "phenom", or "circuit_level"
            d_list - list of distances to run
            l_list - list of elongations to run
            eta_list - list of noise biases to run
            cd_list - list of clifford deformations to run, either "SC", "XZZXonSqu", or "ZXXZonSqu". Not to be used if corr_decoding is True and circuit_data is False.
            corr_list - list of correlation types to run, either "CORR_XZ" or "CORR_ZX". Only to be used if corr_decoding is True and circuit_data is False.
            total_num_shots - int, the total number of shots after averaging. Each SLURM array task will run total_num_shots/reps shots.
            p_list - list of physical error rates to scan over. If None, will be set based on p_th_init_d
            p_th_init_d - dictionary with keys (l, eta, corr_type) or (l, eta, cd_type) and values the initial guess for the threshold. If None, will use a default value based on eta
            pymatch_corr - boolean, whether to use pymatching correlated decoder for circuit data
        out: no output, but will write data to a CSV file for each SLURM array task. Run concat_csv after all tasks are complete to combine the CSV files into output_file.
    """


    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) # will iter over the total slurm array size and points to where you are 
    slurm_array_size = int(os.environ['SLURM_ARRAY_TASK_MAX']) # the size of the slurm array, used to determine how many tasks to run, currently 1000

    print(f"Task ID: {task_id}")
    print(f"SLURM Array Size: {slurm_array_size}")


    if circuit_data and not (corr_decoding and pymatch_corr): # change this to get different data for circuit level plot
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM", cd_type, noise_model)]
            p_list = np.linspace(max(p_th_init - 0.001, 0.0), min(p_th_init + 0.001, 1.0), 40)
        write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type, circuit_data=circuit_data, noise_model=noise_model, cd_type=cd_type, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)
    if circuit_data and (pymatch_corr or corr_decoding):
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id, pymatching corr should be doing an erasure channel this whole time, see what happens
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM_CORR", cd_type,noise_model)] # add the mem type somehow
            p_list = np.linspace(p_th_init - 0.001, p_th_init + 0.001, 40)
        write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type, circuit_data=circuit_data, noise_model=noise_model, cd_type=cd_type, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)
    

    if corr_decoding and not circuit_data: # change this to get different data for eta plot
        l_eta_corr_type_arr = list(itertools.product(l_list, eta_list, corr_list)) # list of tuples (l, eta, corr_type), currently 40
        reps = slurm_array_size//len(l_eta_corr_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_corr_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, corr_type = l_eta_corr_type_arr[ind] # get the l and eta from the task_id
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, corr_type)]
            p_list = np.linspace(p_th_init - 0.03, p_th_init + 0.03, 40)
        num_shots = int(total_num_shots//reps) # number of shots to sample
        cd_type = "SC"
        noise_model = "code_cap"
        print("l,eta,corr_type", l,eta, corr_type)
        write_data(num_shots, d_list, l, p_list, eta, task_id, corr_type, circuit_data=circuit_data, noise_model=noise_model, cd_type=cd_type,  corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)
    
    print("reps", reps)
    print("ind", ind)
    print("num_shots", num_shots)



def get_thresholds_from_data_exactish(num_shots, p_th_init_dict, p_range, output_file):
    """
    Given a dictionary of thresholds, get the thresholds from the data files and add them to the dictionary
    in: num_shots - the number of shots to sample
        p_th_init_dict - a dictionary of initial guesses for the threshold, only the entries you want to make exactish, with keys (l, eta, corr_type)
    out: threshold_d - the updated dictionary of thresholds
    """
    all_thresholds_df = pd.read_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv')
    # print("thresholds_df 1", all_thresholds_df)


    for key in p_th_init_dict.keys():
        # print("key", key)
        l, eta, corr_type, CD_type, noise_model = key
        print("l,eta,corr_type, CD_type, noise_model", l,eta, corr_type, CD_type, noise_model)

        df = pd.read_csv(output_file)
        # threshold_d = {}

        p_th_init = p_th_init_dict[key]
        pop,pcov = get_threshold(df, p_th_init,p_range, l, eta, corr_type, num_shots, CD_type)
        print(p_th_init, pop)
        threshold = pop[0]
        std_error = np.sqrt(np.diag(pcov))[0] # should it be np.sqrt(np.trace(pcov)) instead to get the overall error?
        # threshold_d[key] = threshold
        all_thresholds_df = pd.concat([all_thresholds_df,pd.DataFrame({'l':l,'eta':eta, 'error_type':corr_type, 'CD_type':CD_type, 'noise_model':noise_model, 'pth':threshold, 'stderr':std_error}, index=[0])], ignore_index=True)
        # print("thresholds_df 2", all_thresholds_df)

    all_thresholds_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv', index=False)


#
# for generating a threshold graph for Z/X too 
#

if __name__ == "__main__":

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
                        (6,4,"CORR_ZX"): 0.222,
                        (2,1.5,"CORR_XZ"): 0.152, (2,1.5,"CORR_ZX"):0.130, (2,2.5,"CORR_XZ"):0.131, (2,2.5,"CORR_ZX"):0.118,
                        (2,3.5,"CORR_XZ"): 0.123, (2,3.5,"CORR_ZX"): 0.113, (2,4.5,"CORR_XZ"): 0.118, (2,4.5,"CORR_ZX"): 0.111,
                        (2,6,"CORR_XZ"): 0.114, (2,6,"CORR_ZX"): 0.108, (2,7,"CORR_XZ"): 0.112, (2,7,"CORR_ZX"): 0.107,
                        (3,1.5,"CORR_XZ"): 0.175, (3,1.5,"CORR_ZX"): 0.173, (3,2.5,"CORR_XZ"): 0.174, (3,2.5,"CORR_ZX"): 0.170,
                        (3,3.5,"CORR_XZ"): 0.163, (3,3.5,"CORR_ZX"): 0.160, (3,4.5,"CORR_XZ"): 0.159, (3,4.5,"CORR_ZX"): 0.158,
                        (3,6,"CORR_XZ"): 0.154, (3,6,"CORR_ZX"): 0.152, (3,7,"CORR_XZ"): 0.153, (3,7,"CORR_ZX"): 0.151,
                        (4,1.5,"CORR_XZ"): 0.176, (4,1.5,"CORR_ZX"): 0.177, (4,2.5,"CORR_XZ"): 0.193, (4,2.5,"CORR_ZX"): 0.194,
                        (4,3.5,"CORR_XZ"): 0.194, (4,3.5,"CORR_ZX"): 0.194, (4,4.5,"CORR_XZ"): 0.189, (4,4.5,"CORR_ZX"): 0.189,
                        (4,6,"CORR_XZ"): 0.183, (4,6,"CORR_ZX"): 0.184, (4,7,"CORR_XZ"): 0.181, (4,7,"CORR_ZX"): 0.180,
                        (5,1.5,"CORR_XZ"): 0.169, (5,1.5,"CORR_ZX"): 0.163, (5,2.5,"CORR_XZ"): 0.198, (5,2.5,"CORR_ZX"): 0.201,
                        (5,3.5,"CORR_XZ"): 0.209,(5,3.5,"CORR_ZX"): 0.210, (5,4.5,"CORR_XZ"): 0.209,(5,4.5,"CORR_ZX"): 0.209,
                        (5,6,"CORR_XZ"): 0.203, (5,6,"CORR_ZX"): 0.205, (5,7,"CORR_XZ"): 0.200, (5,7,"CORR_ZX"): 0.202,
                        (6,1.5,"CORR_XZ"): 0.135, (6,1.5,"CORR_ZX"): 0.118, (6,2.5,"CORR_XZ"): 0.20, (6,2.5,"CORR_ZX"): 0.202,
                        (6,3.5,"CORR_XZ"): 0.217, (6,3.5,"CORR_ZX"): 0.224, (6,4.5,"CORR_XZ"): 0.224, (6,4.5,"CORR_ZX"): 0.227,
                        (6,6,"CORR_XZ"): 0.224, (6,6,"CORR_ZX"): 0.227, (6,7,"CORR_XZ"): 0.222, (6,7,"CORR_ZX"): 0.225,
                        (2,1.67, "CORR_XZ"): 0.127, (2,1.67, "CORR_ZX"):0.148, (2,3, "CORR_XZ"):0.127, (2,3, "CORR_ZX"):0.116,
                        (2,4.26, "CORR_XZ"):0.121, (2,4.26, "CORR_ZX"):0.113, (2,5.89, "CORR_XZ"):0.116, (2,5.89, "CORR_ZX"):0.109,
                        (3,1.67, "CORR_XZ"): 0.176, (3,1.67, "CORR_ZX"):0.179, (3,3, "CORR_XZ"):0.169, (3,3, "CORR_ZX"):0.164,
                        (3,4.26, "CORR_XZ"):0.160, (3,4.26, "CORR_ZX"):0.158, (3,5.89, "CORR_XZ"):0.156, (3,5.89, "CORR_ZX"):0.152,
                        (4,1.67, "CORR_XZ"): 0.181, (4,1.67, "CORR_ZX"):0.183, (4,3, "CORR_XZ"):0.196, (4,3, "CORR_ZX"):0.197,
                        (4,4.26, "CORR_XZ"):0.192, (4,4.26, "CORR_ZX"):0.192, (4,5.89, "CORR_XZ"):0.185, (4,5.89, "CORR_ZX"):0.185,
                        (5,1.67, "CORR_XZ"): 0.178, (5,1.67, "CORR_ZX"):0.176, (5,3, "CORR_XZ"):0.206, (5,3, "CORR_ZX"):0.208,
                        (5,4.26, "CORR_XZ"):0.211,(5,4.26, "CORR_ZX"):0.212, (5,5.89, "CORR_XZ"):0.205,(5,5.89, "CORR_ZX"):0.207,
                        (6,1.67, "CORR_XZ"): 0.161, (6,1.67, "CORR_ZX"):0.144, (6,3, "CORR_XZ"): 0.212, (6,3, "CORR_ZX"):0.215,
                        (6,4.26, "CORR_XZ"): 0.225, (6,4.26, "CORR_ZX"):0.226, (6,5.89, "CORR_XZ"): 0.227, (6,5.89, "CORR_ZX"):0.229
                        }

    p_th_init_dict_C_CC = {(2, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.11, (2, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.11,
                         (2, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.17, (2, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.16,
                         (2, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.19, (2, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.19,
                         (2, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.27, (2, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.23,
                         (2, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.29, (2, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.24,
                         (2, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.3, (2, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.22,
                         (2, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (2, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.21,
                         (3, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.15, (3, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.082,
                         (3, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.245, (3, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.105,
                         (3, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.277, (3, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.115,
                         (3, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (3, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.1376,
                         (3, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (3, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (3, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (3, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (3, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (3, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (4, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.186, (4, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.061,
                         (4, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.255, (4, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.098,
                         (4, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.282, (4, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.11,
                         (4, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (4, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (4, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.34, (4, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.136,
                         (4, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (4, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.132,
                         (4, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (4, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (5, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.21, (5, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.055,
                         (5, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.26, (5, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.090,
                         (5, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.287, (5, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.10,
                         (5, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (5, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.127,
                         (5, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.346, (5, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (5, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (5, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (5, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (5, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.132,
                         (6, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.23, (6, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.05,
                         (6, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.265, (6, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.090,
                         (6, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.29, (6, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.10,
                         (6, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (6, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.125,
                         (6, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (6, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (6, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (6, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.13,
                         (6, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (6, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.126,
                         (2,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (2,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (2,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00, (3,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, 
                         (3,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00, (3,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00,
                         (4,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (4,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (4,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00, (5,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, 
                         (5,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00, (5,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00,
                         (6,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (6,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (6,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00
                         }
    
    p_th_init_CL = {
                    (2,0.5,"X_MEM", "SC","circuit_level"):0.00721, (2,0.5,"Z_MEM", "SC","circuit_level"):0.00736, (2,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00728,
                    (4,0.5,"X_MEM","SC", "circuit_level"):0.00935, (4,0.5,"Z_MEM", "SC","circuit_level"):0.00466, (4,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00474,
                    (6,0.5,"X_MEM", "SC","circuit_level"):0.01027, (6,0.5,"Z_MEM", "SC","circuit_level"):0.00354, (6,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00348,
                    (2,5,"X_MEM","SC", "circuit_level"):0.00734, (2,5,"Z_MEM", "SC","circuit_level"):0.01051, (2,5,"TOTAL_MEM","SC", "circuit_level"):0.00745,
                    (4,5,"X_MEM", "SC","circuit_level"):0.01057, (4,5,"Z_MEM", "SC","circuit_level"):0.00648, (4,5,"TOTAL_MEM","SC", "circuit_level"):0.00721,
                    (6,5,"X_MEM", "SC","circuit_level"):0.01219, (6,5,"Z_MEM", "SC","circuit_level"):0.00491, (6,5,"TOTAL_MEM", "SC","circuit_level"):0.00500,
                    (2,10,"X_MEM","SC", "circuit_level"):0.00740, (2,10,"Z_MEM","SC", "circuit_level"):0.01116, (2,10,"TOTAL_MEM", "SC","circuit_level"):0.00747,
                    (4,10,"X_MEM", "SC","circuit_level"):0.01101, (4,10,"Z_MEM", "SC","circuit_level"):0.00679, (4,10,"TOTAL_MEM", "SC","circuit_level"):0.00802,
                    (6,10,"X_MEM", "SC","circuit_level"):0.01279, (6,10,"Z_MEM", "SC","circuit_level"):0.00527, (6,10,"TOTAL_MEM", "SC","circuit_level"):0.00550,
                    (2,25,"X_MEM", "SC","circuit_level"):0.00745, (2,25,"Z_MEM","SC", "circuit_level"):0.01196, (2,25,"TOTAL_MEM", "SC","circuit_level"):0.00753,
                    (4,25,"X_MEM", "SC","circuit_level"):0.01129, (4,25,"Z_MEM", "SC","circuit_level"):0.00726, (4,25,"TOTAL_MEM", "SC","circuit_level"):0.00871,
                    (6,25,"X_MEM", "SC","circuit_level"):0.01329, (6,25,"Z_MEM", "SC","circuit_level"):0.00572, (6,25,"TOTAL_MEM","SC", "circuit_level"):0.00593,
                    (2,50,"X_MEM", "SC","circuit_level"):0.00745, (2,50,"Z_MEM", "SC","circuit_level"):0.01230, (2,50,"TOTAL_MEM", "SC","circuit_level"):0.00757,
                    (4,50,"X_MEM", "SC","circuit_level"):0.01150, (4,50,"Z_MEM", "SC","circuit_level"):0.00751, (4,50,"TOTAL_MEM", "SC","circuit_level"):0.00886,
                    (6,50,"X_MEM","SC", "circuit_level"):0.01346, (6,50,"Z_MEM", "SC","circuit_level"):0.00584, (6,50,"TOTAL_MEM", "SC","circuit_level"):0.00603,
                    (2,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.00734, (2,0.5,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00726, (2,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00732,
                    (4,0.5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00941, (4,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00455, (4,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00458,
                    (6,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01034, (6,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00354, (6,0.5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00352,
                    (2,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00863, (2,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00873, (2,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00873,
                    (4,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.01232, (4,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00548, (4,5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00555,
                    (6,5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01367, (6,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00436, (6,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00441,
                    (2,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.00911, (2,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00913, (2,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00911,
                    (4,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.01314, (4,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00574, (4,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00590,
                    (6,10,"X_MEM", "ZXXZonSqu","circuit_level"):0.01473, (6,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00460, (6,10,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00455,
                    (2,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.00928, (2,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00932, (2,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00935,
                    (4,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01384, (4,25,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00607, (4,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00612,
                    (6,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01570, (6,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00481, (6,25,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00477,
                    (2,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.00949, (2,50,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00954, (2,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00949,
                    (4,50,"X_MEM","ZXXZonSqu", "circuit_level"):0.01411, (4,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00603, (4,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00616,
                    (6,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.01610, (6,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00468, (6,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00493,
                    (3,0.5,"X_MEM","SC", "circuit_level"):0.00835, (3,0.5,"Z_MEM", "SC","circuit_level"):0.00569, (3,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00595,
                    (5,0.5,"X_MEM", "SC","circuit_level"):0.00985, (5,0.5,"Z_MEM", "SC","circuit_level"):0.00411, (5,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00409,
                    (3,5,"X_MEM", "SC","circuit_level"):0.00937, (3,5,"Z_MEM", "SC","circuit_level"):0.00789, (3,5,"TOTAL_MEM","SC", "circuit_level"):0.00873,
                    (5,5,"X_MEM", "SC","circuit_level"):0.01143, (5,5,"Z_MEM", "SC","circuit_level"):0.00565, (5,5,"TOTAL_MEM", "SC","circuit_level"):0.00586,
                    (3,10,"X_MEM", "SC","circuit_level"):0.00964, (3,10,"Z_MEM", "SC","circuit_level"):0.00842, (3,10,"TOTAL_MEM", "SC","circuit_level"):0.00922,
                    (5,10,"X_MEM", "SC","circuit_level"):0.01194, (5,10,"Z_MEM", "SC","circuit_level"):0.00603, (5,10,"TOTAL_MEM", "SC","circuit_level"):0.00645,
                    (3,25,"X_MEM", "SC","circuit_level"):0.00985, (3,25,"Z_MEM", "SC","circuit_level"):0.00899, (3,25,"TOTAL_MEM", "SC","circuit_level"):0.00960,
                    (5,25,"X_MEM", "SC","circuit_level"):0.01249, (5,25,"Z_MEM", "SC","circuit_level"):0.00637, (5,25,"TOTAL_MEM","SC", "circuit_level"):0.00700,
                    (3,50,"X_MEM", "SC","circuit_level"):0.00987, (3,50,"Z_MEM", "SC","circuit_level"):0.00916, (3,50,"TOTAL_MEM", "SC","circuit_level"):0.00970,
                    (5,50,"X_MEM","SC", "circuit_level"):0.01255, (5,50,"Z_MEM", "SC","circuit_level"):0.00654, (5,50,"TOTAL_MEM", "SC","circuit_level"):0.00719,
                    (3,0.5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00852, (3,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00559, (3,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00574,
                    (5,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.00987, (5,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00389, (5,0.5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00394,
                    (3,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.01097, (3,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00669, (3,5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00698,
                    (5,5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01298, (5,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00483, (5,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00481,
                    (3,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.01158, (3,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00688, (3,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00734,
                    (5,10,"X_MEM", "ZXXZonSqu","circuit_level"):0.01390, (5,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00508, (5,10,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00517,
                    (3,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01297, (3,25,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00715, (3,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00759,
                    (5,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01466, (5,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00525, (5,25,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00538,
                    (3,50,"X_MEM","ZXXZonSqu", "circuit_level"):0.01288, (3,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00713, (3,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00764,
                    (5,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.01502, (5,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00529, (5,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00538,
    }
                    
    p_th_init_CL_pycorr = {                # CHAT DATA In the middle

                    # ===================== SC =====================
                    # eta = 0.5
                    (2,0.5,"X_MEM_PY","SC","circuit_level"):0.00852,
                    (2,0.5,"Z_MEM_PY","SC","circuit_level"):0.00871,
                    (2,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00859,

                    (3,0.5,"X_MEM_PY","SC","circuit_level"):0.00897,
                    (3,0.5,"Z_MEM_PY","SC","circuit_level"):0.00721,
                    (3,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00768,

                    (4,0.5,"X_MEM_PY","SC","circuit_level"):0.00949,
                    (4,0.5,"Z_MEM_PY","SC","circuit_level"):0.00620,
                    (4,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00652,

                    (5,0.5,"X_MEM_PY","SC","circuit_level"):0.00996,
                    (5,0.5,"Z_MEM_PY","SC","circuit_level"):0.00546,
                    (5,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00565,

                    (6,0.5,"X_MEM_PY","SC","circuit_level"):0.01034,
                    (6,0.5,"Z_MEM_PY","SC","circuit_level"):0.00485,
                    (6,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00493,


                    # eta = 5
                    (2,5,"X_MEM_PY","SC","circuit_level"):0.00801,
                    (2,5,"Z_MEM_PY","SC","circuit_level"):0.01167,
                    (2,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,5,"X_MEM_PY","SC","circuit_level"):0.01002,
                    (3,5,"Z_MEM_PY","SC","circuit_level"):0.00962,
                    (3,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00989,

                    (4,5,"X_MEM_PY","SC","circuit_level"):0.01097,
                    (4,5,"Z_MEM_PY","SC","circuit_level"):0.00814,
                    (4,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00916,

                    (5,5,"X_MEM_PY","SC","circuit_level"):0.01109,
                    (5,5,"Z_MEM_PY","SC","circuit_level"):0.00719,
                    (5,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00768,

                    (6,5,"X_MEM_PY","SC","circuit_level"):0.01226,
                    (6,5,"Z_MEM_PY","SC","circuit_level"):0.00624,
                    (6,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00606,


                    # eta = 10
                    (2,10,"X_MEM_PY","SC","circuit_level"):0.00863,
                    (2,10,"Z_MEM_PY","SC","circuit_level"):0.01241,
                    (2,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,10,"X_MEM_PY","SC","circuit_level"):0.01029,
                    (3,10,"Z_MEM_PY","SC","circuit_level"):0.01008,
                    (3,10,"TOTAL_MEM_PY","SC","circuit_level"):0.01029,

                    (4,10,"X_MEM_PY","SC","circuit_level"):0.01137,
                    (4,10,"Z_MEM_PY","SC","circuit_level"):0.00861,
                    (4,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00977,

                    (5,10,"X_MEM_PY","SC","circuit_level"):0.01215,
                    (5,10,"Z_MEM_PY","SC","circuit_level"):0.00789,
                    (5,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00852,

                    (6,10,"X_MEM_PY","SC","circuit_level"):0.01285,
                    (6,10,"Z_MEM_PY","SC","circuit_level"):0.00669,
                    (6,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00732,

                    # eta = 25
                    (2,25,"X_MEM_PY","SC","circuit_level"):0.00871,
                    (2,25,"Z_MEM_PY","SC","circuit_level"):0.01306,
                    (2,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,25,"X_MEM_PY","SC","circuit_level"):0.01053,
                    (3,25,"Z_MEM_PY","SC","circuit_level"):0.01076,
                    (3,25,"TOTAL_MEM_PY","SC","circuit_level"):0.01063,

                    (4,25,"X_MEM_PY","SC","circuit_level"):0.01169,
                    (4,25,"Z_MEM_PY","SC","circuit_level"):0.00922,
                    (4,25,"TOTAL_MEM_PY","SC","circuit_level"):0.01044,

                    (5,25,"X_MEM_PY","SC","circuit_level"):0.01257,
                    (5,25,"Z_MEM_PY","SC","circuit_level"):0.00795,
                    (5,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00909,

                    (6,25,"X_MEM_PY","SC","circuit_level"):0.01340,
                    (6,25,"Z_MEM_PY","SC","circuit_level"):0.00726,
                    (6,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00776,

                    # eta = 50
                    (2,50,"X_MEM_PY","SC","circuit_level"):0.00867,
                    (2,50,"Z_MEM_PY","SC","circuit_level"):0.01340,
                    (2,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00865,

                    (3,50,"X_MEM_PY","SC","circuit_level"):0.01061,
                    (3,50,"Z_MEM_PY","SC","circuit_level"):0.01095,
                    (3,50,"TOTAL_MEM_PY","SC","circuit_level"):0.01067,

                    (4,50,"X_MEM_PY","SC","circuit_level"):0.01186,
                    (4,50,"Z_MEM_PY","SC","circuit_level"):0.00939,
                    (4,50,"TOTAL_MEM_PY","SC","circuit_level"):0.01061,

                    (5,50,"X_MEM_PY","SC","circuit_level"):0.01281,
                    (5,50,"Z_MEM_PY","SC","circuit_level"):0.00814,
                    (5,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00935,

                    (6,50,"X_MEM_PY","SC","circuit_level"):0.01363,
                    (6,50,"Z_MEM_PY","SC","circuit_level"):0.00740,
                    (6,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00802,


                    # ===================== ZXXZonSqu =====================
                    # eta = 0.5
                    (2,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00850,
                    (2,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00852,
                    (2,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00959,

                    (3,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00910,
                    (3,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00721,
                    (3,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00751,

                    (4,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00948,
                    (4,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00610,
                    (4,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00635,

                    (5,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01015,
                    (5,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00527,
                    (5,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00540,

                    (6,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01048,
                    (6,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00477,
                    (6,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00496,


                    # eta = 5
                    (2,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00998,
                    (2,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.0100,
                    (2,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00974,

                    (3,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01146,
                    (3,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00823,
                    (3,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00871,

                    (4,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01245,
                    (4,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00678,
                    (4,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00728,

                    (5,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01323,
                    (5,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00614,
                    (5,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00631,

                    (6,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01378,
                    (6,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00557,
                    (6,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00569,

                    # eta = 10 

                    (2,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01038,
                    (2,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01038,
                    (2,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01052,

                    (3,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01205,
                    (3,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00850,
                    (3,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00922,

                    (4,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01325,
                    (4,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00740,
                    (4,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00766,

                    (5,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01401,
                    (5,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00641,
                    (5,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00658,

                    (6,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01479,
                    (6,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00570,
                    (6,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00586,

                    # eta = 25
                    (2,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01076,
                    (2,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01008,
                    (2,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01070,

                    (3,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01251,
                    (3,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00882,
                    (3,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00945,

                    (4,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01397,
                    (4,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00751,
                    (4,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00772,

                    (5,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01487,
                    (5,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00677,
                    (5,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00688,

                    (6,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01574,
                    (6,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00603,
                    (6,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00620,

                    # eta = 50
                    (2,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01082,
                    (2,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01087,
                    (2,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01089,

                    (3,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01285,
                    (3,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00897,
                    (3,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00960,

                    (4,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01420,
                    (4,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00772,
                    (4,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00802,

                    (5,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01536,
                    (5,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00683,
                    (5,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00704,

                    (6,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01629,
                    (6,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00618,
                    (6,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00641 }



                    # (2,0.5,"X_MEM_PY", "SC","circuit_level"):0.00852, (2,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00869, (2,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00859,
                    # (3,0.5,"X_MEM_PY","SC", "circuit_level"):0.00897, (3,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00721, (3,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00768,
                    # (4,0.5,"X_MEM_PY", "SC","circuit_level"):0.00949, (4,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00620, (4,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00652,
                    # (5,0.5,"X_MEM_PY", "SC","circuit_level"):0.00996, (5,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00546, (5,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00,
                    # (6,0.5,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,5,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,5,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,5,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,5,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,5,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,5,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,5,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,5,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,10,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,10,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,10,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,10,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,10,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,10,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,10,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,10,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,10,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,10,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,25,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,25,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,25,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,25,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,25,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,25,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,25,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,25,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,25,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,25,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,50,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,50,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,50,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,50,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,50,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,50,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,50,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,50,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,50,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,50,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,0.5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,0.5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,10,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,10,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,25,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,25,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,50,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,50,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474}

    #### parameters


    circuit_data = True # whether circuit level or code cap data is desired
    corr_decoding = True # whether to get data for correlated decoding (corrxz or corrzx), or circuit level (X/Z mem or X/Z mem py)
        

    # simulation

    # if getting threshold specific data
    # p_th_init = p_th_init_dict[(l,eta,corr_type)]
    # p_th_init = 0.158
    # p_list = np.linspace(p_th_init-0.03, p_th_init + 0.03, 40)

    # otherwise p_list is range of probabilities
    p_list = np.logspace(-2.5, -1.5, 40)
    # p_list = None

    l_list = [2,4,6] # elongation params, do 3 and 5 in another batch
    d_list = [11,13,15,17,19] # code distances
    eta_list = [0.5,5,10,25,50] # noise bias
    cd_list = ["SC", "ZXXZonSqu"] # clifford deformation types
    total_num_shots = 1000 # number of shots 
    corr_type = "TOTAL_MEM_CORR" # which type of correlation to use, depending on the type of decoder. Choose from ['CORR_XZ', 'CORR_ZX', 'TOTAL', 'TOTAL_MEM', 'TOTAL_PY_CORR', 'TOTAL_MEM_CORR']
    error_type = "TOTAL_MEM_CORR" # which type of error to plot
    # num_shots = 66666
    corr_list = ['CORR_XZ', 'CORR_ZX']
    corr_type_list = ['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR']  
    noise_model = "circuit_level"
    py_corr = False # whether to use pymatching correlated decoder for circuit data

    if circuit_data:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data/'
        if noise_model == "circuit_level":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data.csv'
        elif noise_model == "code_cap":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/code_cap_circuit_data.csv'
        elif noise_model == "phenom":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/phenom_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_circuit_data.csv'
    else:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/corr_err_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'


    # run this to get data from the dcc
    get_data_DCC(circuit_data, corr_decoding, noise_model, d_list, l_list, eta_list, cd_list, corr_list, total_num_shots, p_list=None, p_th_init_d=p_th_init_CL_pycorr, pymatch_corr=py_corr)

    # run this once you have data and want to combo it to one csv
    # concat_csv(folder_path, circuit_data)


    # plot the threshold results

    # eta - 0.5, 5, 10, 25, 50, retake for lower ranges at lower eta 
    # l - 2,3,4,5,6
    # no corr
    # num_shots - 30303, total - 1e6
    # d - 11-19 odd 
    # CD_type - SC, ZXXZonSqu



    # params to plot
    # eta = 0.5
    # l = 2
    # curr_num_shots = 52631.0 # the file has 20408 for the 3,5 and 30303 for the 2,4,6
    # noise_model = "circuit_level"
    # CD_type = "ZXXZonSqu"
    # py_corr = True # whether to use pymatching correlated decoder for circuit data
    # corr_decoding = False # whether to get data for correlated decoding using my decoder
    # error_type = "Z_MEM_PY" # which type of error to plot, choose from ['X_MEM', 'Z_MEM', 'TOTAL_MEM', 'TOTAL_PY_MEM', 'TOTAL_MEM_PY_CORR']
    # p_range = 0.001

    



    # df = pd.read_csv(output_file)

    # full_error_plot(df,eta,l,curr_num_shots,noise_model, CD_type, output_file,corr_decoding=corr_decoding, py_corr=py_corr, circuit_level=circuit_data)


    # make a plot for specific thresholds
    # pth0 = p_th_init_CL_pycorr[(l, eta, error_type, CD_type, noise_model)]
    # popt, pcov = get_threshold(df, pth0, p_range, l, eta, error_type, curr_num_shots, CD_type)
    # p_th = popt[0]
    # pth_error = np.sqrt(pcov[0][0])
    # print(p_th, pth_error)
    # get_thresholds_from_data_exactish(curr_num_shots, p_th_init_CL_pycorr,p_range, output_file)
    # eta_df = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv")
    # p_range_df = df[(df['p'] <= pth0 + p_range) & (df["p"] >= pth0 - p_range)]
    # print(len(p_range_df))
    # threshold_plot(df, pth0, p_range, eta, l, curr_num_shots, error_type, CD_type, noise_model, file=output_file, circuit_level=True, py_corr = py_corr, corr_decoding=corr_decoding, loglog=False, averaging=True, show_threshold=True, show_fit=True)
    eta_threshold_plot(eta_df, CD_type,corr_type_list, noise_model)
    # get_thresholds_from_data_exactish(curr_num_shots, p_th_init_CL,p_range, output_file)
    # make eta plot
    # eta_df = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv")
    # corr_type_list = ['TOTAL', "TOTAL_PY_CORR"]
    # eta_threshold_plot(eta_df, "XZZXonSqu", corr_type_list, "code_cap")



