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
from clifford_deformed_cc_circuit import CDCompassCodeCircuit
import itertools
import stim
from compass_code_correlated_error import CorrelatedDecoder


def get_LER_over_cutoff(d,l,eta,mem_type, CD_type, p, num_shots, cutoff):
    circuit_obj = CDCompassCodeCircuit(d, l, eta, mem_type)
    circuit = circuit_obj.make_elongated_circuit_from_parity(before_measure_flip=p,
                                                            before_measure_pauli_channel=0,
                                                            after_clifford_depolarization=p,
                                                            before_round_data_pauli_channel=0,
                                                            between_round_idling_pauli_channel=p,
                                                            idling_dephasing=0,
                                                            CD_type=CD_type)
    corr_decoder = CorrelatedDecoder(eta,d,l,"CORR_XZ", mem_type=mem_type)
    log_errors_corr_gap = corr_decoder.decoding_failures_correlated_gap(circuit, num_shots, mem_type, CD_type, cutoff)
    return sum(log_errors_corr_gap)/num_shots


## Params to run on DCC
num_shots = 1_000_000
l = 2
mem_type = "Z"
CD_type = "SC"
cutoffs = [0,1,3,5,10,15,20]
d_list = [9,11,13]
p_list = [0.001, 0.005, 0.01, 0.015]
eta_list = [0.5,5,50,500]


## Slurm array indices
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

num_d = len(d_list)
num_p = len(p_list)
num_eta = len(eta_list)
num_cutoff = len(cutoffs)

total_jobs = num_d * num_p * num_eta * num_cutoff
if task_id >= total_jobs:
    raise ValueError(f"Task ID {task_id} exceeds total jobs {total_jobs}")

# Flattened indexing:
# eta changes fastest, then p, then d
d_index = task_id // (num_p * num_eta * num_cutoff)
remainder = task_id % (num_p * num_eta * num_cutoff)

p_index = remainder // (num_eta * num_cutoff)
remainder = remainder % (num_eta * num_cutoff)

eta_index = remainder // num_cutoff
cutoff_index = remainder % num_cutoff

d = d_list[d_index]
p = p_list[p_index]
eta = eta_list[eta_index]
cutoff = cutoffs[cutoff_index]

print(f"Running job for d={d}, p={p}, eta={eta}, cutoff={cutoffs}")

# -------------------------
# Run simulation
# -------------------------

ler = get_LER_over_cutoff(
    d=d,
    l=l,
    eta=eta,
    mem_type=mem_type,
    CD_type=CD_type,
    p=p,
    num_shots=num_shots,
    cutoff=cutoff
)

# -------------------------
# Save results
# -------------------------

row = {
    "d": d,
    "l": l,
    "eta": eta,
    "p": p,
    "num_shots": num_shots,
    "mem_type": mem_type,
    "CD_type": CD_type,
    "cutoff": cutoff,
    "LER": ler,
    "slurm_job_id":  os.environ.get("SLURM_JOB_ID", "local"),
    "slurm_array_id": os.environ.get("SLURM_ARRAY_TASK_ID", "local")
}



mem_type = "Z"
CD_type = "SC"


# -------------------------
# Create results directory
# -------------------------

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

p_str = f"{p:.6f}"
filename = f"result_d{d}_eta{eta}_p{p_str}_cutoff{cutoff}.csv"
output_file = os.path.join(results_dir, filename)

df = pd.DataFrame([row])
df.to_csv(output_file, index=False)    

print(f"Saved {output_file}", flush=True)

    