import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from compass_code_correlated_error import concat_csv
from datetime import datetime

csv_file = 'corr_err_data.csv'
# concat_csv('corr_err_data/', csv_file)
df = pd.read_csv(csv_file)


# Input parameters
curr_l = 6
curr_eta = 5.89
curr_num_shots = 10000

prob_scale = {'x': 2*0.5/(1+curr_eta), 'z': (1+2*curr_eta)/(2*(1+curr_eta)), 'corr_z': 1, 'total':1}

# Filter the DataFrame based on the input parameters
filtered_df = df[(df['l'] == curr_l) & (df['eta'] == curr_eta) & (df['num_shots'] == curr_num_shots) 
                # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
                 ]


# Get unique error types and unique d values
error_types = filtered_df['error_type'].unique()
d_values = filtered_df['d'].unique()

# Create a figure with subplots for each error type
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# Plot each error type in a separate subplot
for i, error_type in enumerate(error_types):
    ax = axes[i]
    error_type_df = filtered_df[filtered_df['error_type'] == error_type]
    
    # Plot each d value
    for d in d_values:
        d_df = error_type_df[error_type_df['d'] == d]
        ax.scatter(d_df['p']*prob_scale[error_type], d_df['num_log_errors'], label=f'd={d}')
    
    ax.set_title(f'Error Type: {error_type}')
    ax.set_xlabel('p')
    ax.set_ylabel('num_log_errors')
    ax.legend()

plt.tight_layout()
plt.show()