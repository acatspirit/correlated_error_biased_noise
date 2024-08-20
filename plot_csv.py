import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from compass_code_correlated_error import concat_csv, shots_averaging, full_error_plot
from datetime import datetime

# Input parameters
curr_l = 2
curr_eta = 0.5
curr_num_shots = 10000
arr_len = 100

csv_file = 'corr_err_data.csv'
if not os.listdir('corr_err_data/'):
    df = pd.read_csv(csv_file)
else:
    csv_file = 'corr_err_data.csv'
    concat_csv('corr_err_data/', csv_file)
    df = pd.read_csv(csv_file)

full_error_plot(df, curr_eta, curr_l, curr_num_shots, averaging=True)

