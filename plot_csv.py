import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


l = 6
num_shots = 500000
eta = 5.89
d_list = [11,13,15,17,19]
p_list = np.linspace(0.01, 0.5, 250)
prob_scale = {'x': 2*0.5/(1+eta), 'z': (1+2*eta)/(2*(1+eta)), 'corr_z': 1, 'total':1}


# with open(f"counter_l{l}_shots{num_shots}_d" + "_".join(map(str, d_list)) + ".txt", "r") as f:
#             counter = int(f.read().strip())

# folder = f"l{l}_shots{num_shots}_d" + "_".join(map(str, d_list)) + f"-{counter}"
# # folder = f"l{l}_shots{num_shots}_large_d"
# files = os.listdir(folder)

# dfs = {}
# for file in files:
#     # Construct the full file path
#     file_path = os.path.join(folder, file)
    
#     # Read the CSV file into a DataFrame and append it to the list
#     df = pd.read_csv(file_path, header=None)
#     dfs[file[:-4]] = df


# # Create a figure with two subplots
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# axs = axs.flatten()

# for i, df in enumerate(dfs.values()):
#     for j in range(len(df)):
#         if i < 2:
#             axs[i].plot(p_list*prob_scale[i], df.iloc[j].values, label=f'd = {d_list[j]}')
#         else:
#             axs[i].plot(p_list, df.iloc[j].values, label=f'd = {d_list[j]}')
#     axs[i].set_title(f"{ind_d[i+1]} errors")
#     axs[i].set_xlabel("Physical Error Rate")
#     axs[i].set_ylabel('Logical Error Rate')
#     axs[i].legend()
#     axs[i].grid(True)

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # # Display the figure with subplots
# plt.show()

# Load the CSV file
csv_file_path = 'corr_err_data.csv'
df = pd.read_csv(csv_file_path)

# Input parameters
curr_l = 6
curr_eta = 5.89
curr_num_shots = 1000

# Filter the DataFrame based on the input parameters
filtered_df = df[(df['l'] == curr_l) & (df['eta'] == curr_eta) & (df['num_shots'] == curr_num_shots)]

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
        ax.plot(d_df['p']*prob_scale[error_type], d_df['num_log_errors'], marker='o', label=f'd={d}')
    
    ax.set_title(f'Error Type: {error_type}')
    ax.set_xlabel('p')
    ax.set_ylabel('num_log_errors')
    ax.legend()

plt.tight_layout()
plt.show()