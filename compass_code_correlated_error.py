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
import stim
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

    def bernoulli_prob(self, old_prob, p):
        """ Given an old probability and a new error probability, return the updated probability
            according to the bernoulli formula
        """
        new_prob = old_prob*(1-p) + p*(1 - old_prob)
        return new_prob  



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
    
    
    #
    # Circuit level correlated decoding functions
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
    
    def decompose_dem_instruction_pairwise(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses pairwise decomposition to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to edges with one detector, boundary node value is -1. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.
            PASS IN DEM with DECOMPOSE_ERRORS=FALSE - talk to ken about this


            eg. error(p) D0 D1 L0 -> {p: p, detectors: [(0, 1)], observables: [0]}
                error(p) D0 -> {p: p, detectors: [(0, -1)], observables: []} single detector error gets boundary edge
                error(p) D0 D2 D1 -> {p:p, detectors: [(0, 2), (2, 1)], observables: []}. 
                error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[]} We choose to ignore ^. If we treated the ^ as already decomposing, we would get [(0,2), (3,-1)]
                error(p) D0 D2 D3 L0 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[0]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary with the probability as the key and a list of edges as the value.
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



    def get_joint_prob(self, dem):
        """ Creates an array of joint probabilities representing edges in the DEM. Each entry [E][F] is the joint probability of edges E and detector F. 
            The diagonal entries [E][E] are the marginal probabilities of one graphlike error mechanism. The joint probabilities are calculated using the bernoulli formula for combining 
            probabilities when two detectors share more than one hyperedge.

            :param dem: stim.DetectorErrorModel object. The detector error model of the circuit to be used in decoding.
            :return: joint_probs: dictionary {[edge 1][edge 2]: joint probability} The joint probability matrix. Each cell is the joint probability of two detectors.
        """

        
        joint_probs = {} # each entry is the joint probability of two edges. [E][E] is a marginal probability

        # iterate through each edge in the dem, add hyperedges
        for inst in dem:
            if inst.type == "error":
                decomposed_inst = self.decompose_dem_instruction_pairwise(inst)
                prob_err = decomposed_inst["p"]
                edges = decomposed_inst["detectors"]

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
                for edge in edges:
                    p = joint_probs.get(edge, {}).get(edge, 0)
                    new_p = self.bernoulli_prob(p, prob_err)
                    joint_probs.setdefault(edge, {})[edge] = new_p
                
        return joint_probs 
    
    def get_conditional_prob(self, joint_prob_dict):
        """ Given a joint probability dictionary, calculates the conditional probabilities for each hyperedge. The conditional probability is given by 
            P(A|B) = P(A^B)/P(A)
            Where A and B are edges from decomposed hyperedges. The marginal probability is P(A), and the joint probability is P(A^B). The maximum conditional probability is 
            P_max = 1/(2*eta + 1), derived from the biased pauli channel. Only hyperedge components are present in final dictionary.

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
            for edge_2 in adjacent_edge_dict:
                if edge_1 == edge_2:  # should I exclude the edge I already picked? Pymatching does
                    continue 

                joint_p = joint_prob_dict.get(edge_1, {}).get(edge_2,0)

                # conditional probability calculation. Min taken because weights cannot be negative, and eta=0.5 represents a full erasure channel
                cond_p = min(1/(2*self.eta + 1), joint_p/marginal_p) # how do I do directionality here / I might have to think about it, will this actually work? Dont wanna fully erase edges...?

                cond_prob_dict.setdefault(edge_1, {})[edge_2] = cond_p
        return cond_prob_dict
    

    def edit_dem(self, edges_in_correction, dem, cond_prob_dict):
        """ Given a stim DEM, updates the probabilities in error instructions with detectors given by cond_prob_dict based on detectors fired in correction.
            If a detector edge picked in the correction has a key in cond_prob_dict, it belonged to a hyperedge. The conditional probability then overwrites
            the original DEM probability for that hyperedge. Logical observables are distributed across new error instructions as in the original instruction.
        """
        # get a list of corrected edges from the first round
        edges_in_correction = [tuple(sorted(edge)) for edge in edges_in_correction]

        # check to make sure edges aren't used twice, leave only the max error probability for each or bernoulli xor?
        # do I need to make sure error mechanisms aren't repeated? - make it a set?

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

    def decoding_failures_correlated_circuit_level(self, circuit, shots):
        """
        Finds the number of logical errors given a circuit using correlated decoding. Uses pymatching's correlated decoding approach, inspired by
        papers cited in the README.
        :param circuit: stim.Circuit object, the circuit to decode
        :param p: physical error rate
        :param shots: number of shots to sample
        :return: number of logical errors
        """

        # 
        # Get the edge data for correlated decoding
        #

        # get the DEM and decompose the errors, get the matching graph
        dem = circuit.detector_error_model(decompose_errors=True) 
        matchgraph = Matching.from_detector_error_model(dem, enable_correlations=False)

        # get the joint probabilities table of the dem hyperedges
        joint_prob_dict = self.get_joint_prob(dem)
        
        # calculate the conditional probabilities based on joint probablities and marginal probabilities 
        cond_prob_dict = self.get_conditional_prob(joint_prob_dict)

        
        
        #
        # Decode the circuit
        #
        
        # first round of decoding
        # get the syndromes and observable flips
        sampler = circuit.compile_detector_sampler(seed=42)
        syndrome, observable_flips = sampler.sample(shots, separate_observables=True)

        # from eva
        # change the logicals so that there is an observable for each qubit, change back to the code cap case to check whether the real logical flipped

        corrections = np.zeros((shots, 2)) # largest fault id is 1, len of correction = 2
        for i in range(shots):
            edges_in_correction = matchgraph.decode_to_edges_array(syndrome[i])
            # update weights based on conditional probabilities
            updated_dem = self.edit_dem(edges_in_correction, dem, cond_prob_dict)

            # second round of decoding with updated weights
            matching_corr = Matching.from_detector_error_model(updated_dem, enable_correlations=False)
            corrections[i] = matching_corr.decode(syndrome[i])
        
        # calculate the number of logical errors
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
        sampler = circuit.compile_detector_sampler(seed=42)
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matching.decode_batch(detection_events)
        
        
        num_errors_array = np.zeros(num_shots)
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors_array[shot] = 1
        return num_errors_array

    def get_num_log_errors_DEM(self, circuit, num_shots, enable_corr, enable_pymatch_corr):
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
            log_errors_array = self.decoding_failures_correlated_circuit_level(circuit, num_shots)

        
        else: # no correlated decoding or pymatching correlated decoding
            dem = circuit.detector_error_model(decompose_errors=enable_pymatch_corr)
            matchgraph = Matching.from_detector_error_model(dem,enable_correlations=enable_pymatch_corr)
            sampler = circuit.compile_detector_sampler(seed=42)
            syndrome, observable_flips = sampler.sample(num_shots, separate_observables=True)
            predictions = matchgraph.decode_batch(syndrome, enable_correlations=enable_pymatch_corr)
            log_errors_array = np.any(np.array(observable_flips) != np.array(predictions), axis=1)
        
        return log_errors_array

    def get_log_error_circuit_level(self, p_list, meas_type, num_shots, noise_model="code_cap", cd_type="SC", corr_decoding= False, pymatch_corr = False):
        """
        Get the logical error rate for a list of physical error rates of gates at the circuit level
        :param p_list: list of p values
        :param meas_type: type of memory experiment(X or Z), stabilizers measured
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
                circuit = circuit_obj.make_elongated_circuit_from_parity(p,0,0,p,0,0,CD_type=cd_type) # check the plots that matched pymatching to get error model right, before meas flip and data qubit pauli between rounds
            elif noise_model == "circuit_level":
                circuit = circuit_obj.make_elongated_circuit_from_parity(p,0,p,0,p,0,CD_type=cd_type)
            else:
                raise ValueError("Invalid noise model. Choose either 'code_cap', 'phenom', or 'circuit_level'.")
            
            log_errors_array = self.get_num_log_errors_DEM(circuit, num_shots, corr_decoding, pymatch_corr)
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

def get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data, noise_model="code_cap", cd_type="SC", corr_decoding=False, pymatch_corr=False):
    """ Generate logical error rates for x,z, correlatex z, and total errors
        via MC sim in decoding_failures_correlated and add it to a shared pandas df
        
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
        
        out: a pandas df recording the logical error rate with all corresponding params

    """
    # print(f"in get data,  l = {l}, eta = {eta}, corr_type = {corr_type}, num_shots = {num_shots}, noise_model = {noise_model}, cd_type = {cd_type}")
    err_type = {0:"X", 1:"Z", 2:corr_type, 3:"TOTAL"}
    if circuit_data:
        data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "noise_model": [], "CD_type":[], "num_log_errors":[], "time_stamp":[]}
    else:
        data_dict = {"d":[], "num_shots":[], "p":[], "l": [], "eta":[], "error_type":[], "num_log_errors":[], "time_stamp":[]}
    data = pd.DataFrame(data_dict)

    for d in d_list:
        if circuit_data:
            # print("running circuit data")
            
                # circuit_x = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "X")
                # circuit_z = cc_circuit.CDCompassCodeCircuit(d, l, eta, [0.003, 0.001, p], "Z")
    
            decoder = CorrelatedDecoder(eta, d, l, corr_type)
            log_errors_z_array = decoder.get_log_error_circuit_level(p_list, "Z", num_shots, noise_model, cd_type, corr_decoding, pymatch_corr) # get the Z logical errors from Z memory experiment, X errors
            log_errors_x_array = decoder.get_log_error_circuit_level(p_list, "X", num_shots, noise_model, cd_type, corr_decoding, pymatch_corr) # get the X logical errors from X memory experiment, Z errors
            log_errors_z = np.sum(log_errors_z_array, axis=1) 
            log_errors_x = np.sum(log_errors_x_array, axis=1)        
            log_errors_total = np.sum(np.logical_xor(log_errors_x_array, log_errors_z_array), axis=1)



            for i,log_error in enumerate(log_errors_x):
                if pymatch_corr:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                elif corr_decoding:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                else:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"X_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)

            for i,log_error in enumerate(log_errors_z):
                if pymatch_corr:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                elif corr_decoding:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                else:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"Z_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)

            for i,log_error in enumerate(log_errors_total):
                if pymatch_corr:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM_PY", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                elif corr_decoding:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM_CORR", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                else:
                    curr_row = {"d":d, "num_shots":num_shots, "p":p_list[i], "l": l, "eta":eta, "error_type":"TOTAL_MEM", "noise_model": noise_model, "CD_type":cd_type, "num_log_errors":log_error/num_shots, "time_stamp":datetime.now()}
                
                data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
            
            

        else:
            decoder = CorrelatedDecoder(eta, d, l, corr_type)

            for p in p_list:
                errors = decoder.decoding_failures_correlated(p, num_shots)
                for i in range(len(errors)):
                    curr_row = {"d":d, "num_shots":num_shots, "p":p, "l": l, "eta":eta, "error_type":err_type[i], "num_log_errors":errors[i]/num_shots, "time_stamp":datetime.now()}
                    data = pd.concat([data, pd.DataFrame([curr_row])], ignore_index=True)
    return data


def shots_averaging(num_shots, l, eta, err_type, in_df, CD_type, file):
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
        data = in_data[(in_data['num_shots'] == num_shots) & (in_data['l'] == l) &(in_data['eta'] == eta) & (in_data['error_type'] == err_type) & (in_data['CD_type'] == CD_type)]
    else:
        data = in_df
    data_mean = data.groupby('p', as_index=False)['num_log_errors'].mean()
    return data_mean



def write_data(num_shots, d_list, l, p_list, eta, ID, corr_type, circuit_data, noise_model="code_cap", cd_type="SC", corr_decoding=False, pymatch_corr=False):
    """ Writes data from pandas df to a csv file, for use with SLURM arrays. Generates data for each slurm output on a CSV
        in: num_shots - the number of MC iterations
            l - the integer repition of the compass code
            eta - the float bias ratio of the error model
            p_list - array of probabilities to scan
            d_list - the distances of compass code to scan
            ID - SLURM input task_ID number, corresponds to which array element we run
    """
    # print(f"in write data, ID = {ID}, l = {l}, eta = {eta}, corr_type = {corr_type}, num_shots = {num_shots}, noise_model = {noise_model}, cd_type = {cd_type}")
    data = get_data(num_shots, d_list, l, p_list, eta, corr_type, circuit_data, noise_model=noise_model, cd_type=cd_type, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)
    if circuit_data:
        if pymatch_corr:
            data_file = f'circuit_data/py_corr_{ID}.csv'
            if not os.path.exists('circuit_data/'):
                os.mkdir('circuit_data')
        else:
            data_file = f'circuit_data/circuit_level_{ID}.csv'
            if not os.path.exists('circuit_data/'):
                os.mkdir('circuit_data')
    else:
        data_file = f'corr_err_data/code_cap_{ID}.csv'
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

def threshold_plot(full_df, p_th0, p_range, curr_eta, curr_l, curr_num_shots, corr_type, file, circuit_level=False, py_corr = False, corr_decoding=False, loglog=False, averaging=True, show_threshold=True):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['p'] > p_th0 - p_range)&(full_df['p'] < p_th0 + p_range)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)]
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


def eta_threshold_plot(eta_df, cd_type, corr_type_list, noise_model):
    """Make a single figure with a 2-column grid of subplots.
    Each row corresponds to a different `l`, with CORR_XZ on left and CORR_ZX on right.
    """
    # print(eta_df)
    # print("Unique cd_type in df:", eta_df['cd_type'].unique())
    # print("cd_type being filtered for:", repr(cd_type))
    # print("Unique noise_model in df:", eta_df['noise_model'].unique())
    # print("noise_model being filtered for:", repr(noise_model))
    eta_df['cd_type'] = eta_df['cd_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()
    # print(cd_type, noise_model)
    df = eta_df[(eta_df['cd_type'] == cd_type) &
                (eta_df['noise_model'] == noise_model)]
    # print(df)
    l_values = sorted(df['l'].unique())
    num_rows = len(l_values)
    num_cols = len(corr_type_list)

    # Set up colors
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, num_rows)
    l_colors = [cmap(val) for val in color_values]

    # Create figure and 2-column grid
    # Create figure and subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 2.5 * num_rows), sharex=True, sharey=True)

    # Make axes 2D for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes[np.newaxis, :]
    elif num_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, l in enumerate(l_values):
        for col_idx, error_type in enumerate(corr_type_list):
            ax = axes[row_idx, col_idx]
            mask = (
                (df['l'] == l) &
                (df['error_type'] == error_type)
            )
            df_filtered = df[mask].sort_values(by='eta')

            eta_vals = df_filtered['eta'].to_numpy()
            pth_list = df_filtered['pth'].to_numpy()
            pth_error_list = df_filtered['stderr'].to_numpy()

            ax.errorbar(
                eta_vals, pth_list, yerr=pth_error_list,
                label=f'l = {l}', color=l_colors[row_idx],
                marker='o', capsize=5
            )

            if row_idx == 0:
                ax.set_title(f"{error_type}, Deformation {cd_type}", fontsize=16)

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
    prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'CORR_ZX':1, 'TOTAL':1, 'TOTAL_MEM':1, 'X_MEM':  1, 'Z_MEM': 1, 'TOTAL_MEM_PY':1, 'X_MEM_PY':1, 'Z_MEM_PY':1} # TOTAL_MEM 4/3 factor of total mem is due to code_cap pauli channel scalling factor in stim, remove this?
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


    if circuit_data and not corr_decoding: # change this to get different data for circuit level plot
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, cd_type)]
            p_list = np.linspace(p_th_init - 0.03, p_th_init + 0.03, 40)
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
            p_th_init = p_th_init_d[(l, eta, cd_type)]
            p_list = np.linspace(p_th_init - 0.03, p_th_init + 0.03, 40)
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



def get_thresholds_from_data_exactish(num_shots, p_th_init_dict):
    """
    Given a dictionary of thresholds, get the thresholds from the data files and add them to the dictionary
    in: num_shots - the number of shots to sample
        p_th_init_dict - a dictionary of initial guesses for the threshold, only the entries you want to make exactish, with keys (l, eta, corr_type)
    out: threshold_d - the updated dictionary of thresholds
    """
    all_thresholds_df = pd.read_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv')
    threshold_d = {}

    # added some stuff for total thresholds that shouldn't be widely used
    # p_min_dict = {}
    # corr_pair = ["CORR_XZ","CORR_ZX"]
    # for key in p_th_init_dict.keys():
    #     l, eta, corr_type = key
    #     alt_corr_type = corr_pair[(corr_pair.index(corr_type) + 1)%2]
    #     threshold_inits = [p_th_init_dict[key],p_th_init_dict[l,eta,alt_corr_type]]
    #     p_min_dict[l,eta, corr_type if threshold_inits.index(min(threshold_inits)) == 0 else alt_corr_type] = min(threshold_inits)

    # for key in p_min_dict.keys():
    #     l, eta, corr_type = key
    #     print("l,eta, corr_type", l,eta, corr_type)

    #     if corr_type == "CORR_ZX":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
    #     df = pd.read_csv(output_file)
    #     # threshold_d = {}

    #     p_th_init = p_min_dict[key]
    #     threshold,std_error = get_threshold(df, p_th_init, 0.03, l, eta, corr_type, num_shots)
    #     threshold_d[key] = threshold
    #     all_thresholds_df = pd.concat([all_thresholds_df,pd.DataFrame({'l':l,'eta':eta, 'error_type':"TOTAL", 'pth':threshold, 'stderr':std_error}, index=[0])], ignore_index=True)

    for key in p_th_init_dict.keys():
        l, eta, corr_type = key
        print("l,eta,corr_type", l,eta, corr_type)

        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
        df = pd.read_csv(output_file)
        # threshold_d = {}

        p_th_init = p_th_init_dict[key]
        threshold,std_error = get_threshold(df, p_th_init, 0.03, l, eta, corr_type, num_shots)
        threshold_d[key] = threshold
        all_thresholds_df = pd.concat([all_thresholds_df,pd.DataFrame({'l':l,'eta':eta, 'error_type':corr_type, 'pth':threshold, 'stderr':std_error}, index=[0])], ignore_index=True)
    

    all_thresholds_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv', index=False)


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

    p_th_init_dict_CL = {(2, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.11, (2, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.11,
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
    
    #### parameters


    circuit_data = True # whether circuit level or code cap data is desired
    corr_decoding = True # whether to get data for correlated decoding (corrxz or corrzx), or circuit level (X/Z mem or X/Z mem py)
        

    # simulation

    # if getting threshold specific data
    # p_th_init = p_th_init_dict[(l,eta,corr_type)]
    # p_th_init = 0.158
    # p_list = np.linspace(p_th_init-0.03, p_th_init + 0.03, 40)

    # otherwise p_list is range of probabilities
    p_list = np.linspace(0.05, 0.4, 40)

    l_list = [2,3,4,5,6] # elongation params
    d_list = [11,13,15,17,19] # code distances
    eta_list = [0.5,5,10,25,50] # noise bias
    cd_list = ["SC","XZZXonSqu", "ZXXZonSqu"] # clifford deformation types
    total_num_shots = 1e4 # number of shots 
    corr_type = "TOTAL_MEM" # which type of correlation to use, depending on the type of decoder. Choose from ['CORR_XZ', 'CORR_ZX', 'TOTAL', 'TOTAL_MEM', 'TOTAL_PY_CORR', 'TOTAL_MEM_CORR']
    error_type = "TOTAL_MEM" # which type of error to plot
    # num_shots = 66666
    corr_list = ['CORR_XZ', 'CORR_ZX']
    corr_type_list = ['TOTAL']  
    noise_model = "code_cap"
    py_corr = False # whether to use pymatching correlated decoder for circuit data

    if circuit_data:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data/'

        output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_circuit_data.csv'
    else:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/corr_err_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'



    # run this to get data from the dcc
    get_data_DCC(circuit_data, corr_decoding, noise_model, d_list, l_list, eta_list, cd_list, corr_list, total_num_shots, p_list=p_list, p_th_init_d=None, pymatch_corr=py_corr)

    # run this once you have data and want to combo it to one csv
    # concat_csv(folder_path, circuit_data)


    # plot the threshold results

    # eta - 0.5, 10, 25, 50, retake for lower ranges at lower eta 
    # l - 2-6 
    # no corr
    # code cap - only one shot, asymmetric pauli channel (before measuremnets circuit) on data
    # preparing the circuit wrong with a probability p is how to think of it
    # write down which one matches


    # params to plot
    # eta = 50
    # l = 6
    # curr_num_shots = 26315.0
    # noise_model = "code_cap"
    # CD_type = "XZZXonSqu"
    # py_corr = False # whether to use pymatching correlated decoder for circuit data


    # df = pd.read_csv(output_file)
    # full_error_plot(df,eta,l,curr_num_shots,noise_model, CD_type, output_file,corr_decoding=corr_decoding, py_corr=py_corr, circuit_level=circuit_data)


    # make eta plot
    # eta_df = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv")
    # corr_type_list = ['TOTAL', "TOTAL_PY_CORR"]
    # eta_threshold_plot(eta_df, "XZZXonSqu", corr_type_list, "code_cap")



