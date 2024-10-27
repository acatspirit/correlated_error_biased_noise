# from typing import Callable, List
import random
from operator import itemgetter
import numpy as np
from pymatching import Matching
from scipy.sparse import csr_matrix
from CompassCodes import *


def clifford_vector_trans_vectorised(xnoise_i, znoise_i, CD_data):
    """
    Inputs:
        xvec_i: Input binary vector with X information (could be a noise vector, state vector, operator)
        zvec_i: Input binary vector with Z information (could be a noise vector, state vector, operator)
        CD_data: dictionary with keys being the qubit position in the vector and the value is the type of
            Clifford applied represented by 0 (Identity), 1 (H_YZ), and 2 (Hadamard H_XZ).
    Outputs:
        xvec: Transformed xvec_i
        zvec: Transformed zvec_i
    """

    # Transform bits
    xnoise = xnoise_i.copy()
    znoise = znoise_i.copy()

    for q, CD in CD_data.items():
        if CD == 2:  # Switch X and Z
            xnoise[:, q] = znoise_i[:, q]
            znoise[:, q] = xnoise_i[:, q]
        elif CD == 1:  # Switch Y and Z
            xnoise[:, q] = (znoise_i[:, q] + xnoise_i[:, q]) % 2 # Q - why are the Y and regular H different conditions

    return xnoise, znoise


def Correction_Clifford_vectorised(xnoise_i, znoise_i, g, mX, mZ, Hx, Hz, CD_data):
    """
    Corrects input noise vectors according to the Clifford deformation specified in dictionary CD_data.
    Input:
        xnoise_i: Input X noise that Clifford Deformed code will see
        znoise_i: Input Z noise that Clifford Deformed code will see
        g: Pre-deformed code class object that will be used to correct errors. Contains stabilizer information
        that is not deformed yet.
        CD_data: Clifford Deformation information for each qubit
    Output:
        xerror: Overall X error on deformed code after correction
        zerror: Overall Z error on deformed code after correction
    """
    # We assume that the incoming noise is detected by Clifford deformed code, so we undo it here to
    # use CSS-type decoding
    # logX, logZ = g.logicals['Z'], g.logicals['X']  # Logical operators
    # L = len(g.qbit_dict)  # Number of data qubits

    # Transform bits
    xnoise, znoise = clifford_vector_trans_vectorised(xnoise_i, znoise_i, CD_data)
    # Assume input are already in CSS frame, so inputs are xnoise, znoise instead of
    # xnoise_i and znoise_i

    # Obtain syndrome and correction
    zsynd = (znoise @ Hx.T) % 2
    zcorr = mX.decode_batch(zsynd)

    xsynd = (xnoise @ Hz.T) % 2
    xcorr = mZ.decode_batch(xsynd)

    return xcorr, zcorr


def num_decoding_failures_CD_vectorised(g, mX, mZ, px, pz, num_trials, cd_data):
    """
    This function counts the logical errors on a Clifford deformed code.
    Inputs:
        g: Code Class object (not deformed)
        px: Vector whose components "i" are probability of X error on qubit "i"
        pz: Vector whose components "i" are probability of Z error on qubit "i"
        num_trials: Number of times to do error correction
        CD_data: Dictionary with Clifford Deformation data
    Outputs:
        num_errors: Total number of logical errors
        num_xerrors: Total number of X logical errors
        num_zerrors: Total number of Z logical errors
    """
    num_errors, num_zerrors, num_xerrors = 0, 0, 0
    # Transforming logical operators
    logX, logZ = np.array([g.logicals["X"]]), np.array([g.logicals["Z"]])
    num_qbits = len(g.qbit_dict)
    # logX_xpart, logX_zpart = Clifford_vectorTrans_vectorised(logX, np.zeros(num_qbits), CD_data)
    # logZ_xpart, logZ_zpart = Clifford_vectorTrans_vectorised(np.zeros(num_qbits), logZ, CD_data)
    # Make matching graphs with appropriately weighted edges
    # mX, mZ = g.make_graph('X'), g.make_graph('Z')  # graph objects
    # Parity Check Matrices
    Hx, Hz = g.H["X"].todense(), g.H["Z"].todense()  # Parity Check matrices

    # Initial noise
    # Noise vector: Each qubit can have no error (0), X error (1), Z error (2), or Y error (3)
    noise = (
        np.random.choice(
            np.arange(4),
            p=[1 - (2 * px + pz), px, pz, px],
            size=(num_trials, num_qbits),
        )
    ).astype(np.uint8)

    # Split into X and Z noise vectors from noise: i-> x z, 0-> 0 0, 1-> 1 0, 2-> 0 1, 3-> 1 1
    znoise_i = np.floor(noise.copy() / 2).astype(np.uint8)
    xnoise_i = noise % 2

    xnoise, znoise = clifford_vector_trans_vectorised(xnoise_i, znoise_i, cd_data)
    # Assume input are already in CSS frame, so inputs are xnoise, znoise instead of
    # xnoise_i and znoise_i

    # Obtain syndrome and correction
    zsynd = (znoise @ Hx.T) % 2
    zcorr = mX.decode_batch(zsynd)

    xsynd = (xnoise @ Hz.T) % 2
    xcorr = mZ.decode_batch(xsynd)

    # Count Errors
    zobs = (znoise @ logX.T) % 2
    xobs = (xnoise @ logZ.T) % 2
    z_predobs = (zcorr @ logX.T) % 2
    x_predobs = (xcorr @ logZ.T) % 2

    num_zerrors = np.sum(np.any(zobs != z_predobs, axis=1))
    num_xerrors = np.sum(np.any(xobs != x_predobs, axis=1))
    num_errors = num_xerrors + num_zerrors

    return num_errors, num_xerrors, num_zerrors


def CDonCompassCode_Simulate_vectorised(Ls, bias, num_trials, ps, rounds = 1, **kwargs):
    """
    This function calculates logical error rate of compass code.
    Input:
        Ls: distances of compass code
        bias: Z bias, depolarizing is 0.5
        num_trials: Number of shots
        ps: list of physical error rates
        **kwargs: Must include either l or p_blue & p_red for compass code creation. Must also include the Clifford
        deformation type ('XZZX' or 'XY') or P_ZX and P_ZY which give probabilities of applying H_ZX and H_ZY respectively.

    Output:
        log_errors_all_L: A list with logical error rates. First index corresponds to L, second index corresponds to p.
    """
    # np.random.seed(2)

    log_errors_all_L = []
    log_xerrors_all_L = []
    log_zerrors_all_L = []
    
    for L in Ls:
        print("Simulating L = {}...".format(L))
        g = CompassCode(L, **kwargs)
        kwargs["size"] = L
        num_qbits = len(g.qbit_dict)
        log_errors = []
        xlog_errors = []
        zlog_errors = []
        CD_data = CD_data_func(range(num_qbits), **kwargs)
        for p in ps:
            num_errors = 0
            num_xerrors = 0
            num_zerrors = 0
            for round in range(1,rounds+1):
                pz = bias * p / (1 + bias)
                px = (p - pz) / 2
                pxs = np.ones(num_qbits) * px
                pzs = np.ones(num_qbits) * pz
                pxs_copy = pxs.copy()
                pzs_copy = pzs.copy()

                # Clifford Deformations change the weights
                for i in range(num_qbits):
                    if CD_data[i] == 0:
                        pass
                    elif CD_data[i] == 1:
                        pxs[i] = pxs_copy[i]
                        pzs[i] = pxs_copy[i]
                    elif CD_data[i] == 2:
                        pxs[i] = pzs_copy[i]
                        pzs[i] = pxs_copy[i]

                # g.add_weight(px=pxs, pz=pzs) # Insert this if you want to use the make_graph method, else we use make_graph_withweights
                mX, mZ = (
                    g.make_graph_withweights("X", ps=pzs),
                    g.make_graph_withweights("Z", ps=pxs),
                )

                num_errors_out, num_xerrors_out, num_zerrors_out = num_decoding_failures_CD_vectorised(
                    g, mX, mZ, px, pz, num_trials, CD_data
                )
                num_errors += num_errors_out
                num_xerrors += num_xerrors_out
                num_zerrors += num_zerrors_out
                
            log_errors.append(num_errors / (num_trials*round))
            xlog_errors.append(num_xerrors / (num_trials*round))
            zlog_errors.append(num_zerrors / (num_trials*round))

        log_errors_all_L.append(np.array(log_errors))
        log_xerrors_all_L.append(np.array(xlog_errors))
        log_zerrors_all_L.append(np.array(zlog_errors))

    return log_errors_all_L, log_xerrors_all_L, log_zerrors_all_L