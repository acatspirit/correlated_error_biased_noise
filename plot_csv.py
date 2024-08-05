import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


l = 6
num_shots = 1000000
eta = 5.89
d_list = [7,9,11,13,15]
p_list = np.linspace(0.01, 0.5, 20)
prob_scale = [2*0.5/(1+eta), (1+2*eta)/(2*(1+eta))]
ind_d = {1:'x', 2:'z', 3:'corr_z', 4:'total'}
folder = f"l{l}_shots{num_shots}_large_d"
files = os.listdir(folder)

dfs = {}
for file in files:
    # Construct the full file path
    file_path = os.path.join(folder, file)
    
    # Read the CSV file into a DataFrame and append it to the list
    df = pd.read_csv(file_path, header=None)
    dfs[file[:-4]] = df


# Create a figure with two subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, df in enumerate(dfs.values()):
    for j in range(len(df)):
        if i < 2:
            axs[i].plot(p_list*prob_scale[i], df.iloc[j].values, label=f'd = {d_list[j]}')
        else:
            axs[i].plot(p_list, df.iloc[j].values, label=f'd = {d_list[j]}')
    axs[i].set_title(f"{ind_d[i+1]} errors")
    axs[i].set_xlabel("Physical Error Rate")
    axs[i].set_ylabel('Logical Error Rate')
    axs[i].legend()
    axs[i].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# # Display the figure with subplots
plt.show()