import os
import sys
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from pymatching import Matching

from clifford_deformed_cc_circuit import CDCompassCodeCircuit
from compass_code_correlated_error import CorrelatedDecoder


def get_circuit(d, p, l=2, eta=0.5, basis='X', CD_type="SC"):
    return CDCompassCodeCircuit(
        d=d,
        l=l,
        eta=eta,
        mem_type=basis
    ).make_elongated_circuit_from_parity(
        before_measure_flip=p,
        before_measure_pauli_channel=0,
        after_clifford_depolarization=p,
        before_round_data_pauli_channel=0,
        between_round_idling_pauli_channel=p,
        idling_dephasing=0,
        CD_type=CD_type
    )


def run_one_chunk(circuit, num_shots, decoder, basis='X', CD_type="SC"):
    """
    Runs one chunk of shots and returns counts, not just rates.
    Returning counts makes aggregation exact.
    """
    sampler = circuit.compile_detector_sampler()
    dem = circuit.detector_error_model(
        decompose_errors=True,
        flatten_loops=True,
        approximate_disjoint_errors=True
    )

    dets, obs = sampler.sample(num_shots, separate_observables=True)

    matching_py_corr = Matching.from_detector_error_model(
        dem,
        enable_correlations=True
    )
    predictions_py_corr = matching_py_corr.decode_batch(
        dets,
        enable_correlations=True
    )

    num_logical_errors_py_corr = int(
        np.sum(np.any(np.array(obs) != np.array(predictions_py_corr), axis=1))
    )

    # Your decoder returns failures per shot; sum them into a count
    num_logical_errors_my_corr = int(
        np.sum(
            decoder.decoding_failures_correlated_circuit_level(
                circuit,
                num_shots,
                basis,
                CD_type
            )
        )
    )

    return num_logical_errors_py_corr, num_logical_errors_my_corr


def make_param_list(d_list, p_list, eta_list):
    params = []
    for d in d_list:
        for eta in eta_list:
            for p in p_list:
                params.append((d, p, eta))
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None,
                        help="Usually supplied from SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--shots_per_chunk", type=int, required=True)
    parser.add_argument("--num_chunks", type=int, required=True,
                        help="How many chunks per (p,eta).")
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--l", type=int, default=3)
    parser.add_argument("--basis", type=str, default="X")
    parser.add_argument("--CD_type", type=str, default="ZXXZonSqu")
    parser.add_argument("--d", type=int, default=9)

    # You can hardcode these or expose them via CLI if you want
    args = parser.parse_args()

    # -------------------------
    # Parameter lists
    # -------------------------
    p_list = np.logspace(-2.5, -2.1, 3)
    eta_list = [0.5, 1, 5]

    params = make_param_list( p_list, eta_list)
    num_param_sets = len(params)
    total_jobs = num_param_sets * args.num_chunks

    # Resolve task_id
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    if task_id < 0 or task_id >= total_jobs:
        raise ValueError(
            f"task_id={task_id} is out of range for total_jobs={total_jobs}"
        )

    # -------------------------
    # Decode task_id into:
    #   param_index, chunk_index
    # -------------------------
    param_index = task_id // args.num_chunks
    chunk_index = task_id % args.num_chunks

    p, eta = params[param_index]

    os.makedirs(args.outdir, exist_ok=True)

    # One file per (d,p,eta,chunk)
    filename = (
        f"chunk_d{args.d}_eta{eta}_p{p:.8f}_l{args.l}_"
        f"{args.CD_type}_{args.basis}_chunk{chunk_index:06d}.csv"
    )
    filepath = os.path.join(args.outdir, filename)

    # Resume protection: skip if file already exists
    if os.path.exists(filepath):
        print(f"[SKIP] File already exists: {filepath}")
        return

    print("=" * 80)
    print(f"Starting task_id={task_id}")
    print(f"d={args.d}, p={p}, eta={eta}, l={args.l}, basis={args.basis}, CD_type={args.CD_type}")
    print(f"chunk_index={chunk_index}, shots_per_chunk={args.shots_per_chunk}")
    print(f"output={filepath}")
    print("=" * 80)
    sys.stdout.flush()

    start_time = datetime.now()

    circuit = get_circuit(
        d=args.d,
        p=p,
        l=args.l,
        eta=eta,
        basis=args.basis,
        CD_type=args.CD_type
    )

    decoder = CorrelatedDecoder(
        eta=eta,
        d=args.d,
        l=args.l,
        corr_type="CORR_XZ",
        mem_type=args.basis
    )

    num_logical_errors_py_corr, num_logical_errors_my_corr = run_one_chunk(
        circuit=circuit,
        num_shots=args.shots_per_chunk,
        decoder=decoder,
        basis=args.basis,
        CD_type=args.CD_type
    )

    end_time = datetime.now()
    elapsed_seconds = (end_time - start_time).total_seconds()

    row = {
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "task_id": task_id,
        "chunk_index": chunk_index,
        "l": args.l,
        "d": args.d,
        "CD_type": args.CD_type,
        "mem_type": args.basis,
        "shots_in_chunk": args.shots_per_chunk,
        "eta": eta,
        "p": p,
        "num_logical_errors_pycorr": num_logical_errors_py_corr,
        "num_logical_errors_my_corr": num_logical_errors_my_corr,
        "ler_pycorr": num_logical_errors_py_corr / args.shots_per_chunk,
        "ler_my_corr": num_logical_errors_my_corr / args.shots_per_chunk
    }

    pd.DataFrame([row]).to_csv(filepath, index=False)

    print(f"[DONE] Wrote {filepath}")
    print(f"Elapsed: {elapsed_seconds:.2f} s")
    sys.stdout.flush()


if __name__ == "__main__":
    main()