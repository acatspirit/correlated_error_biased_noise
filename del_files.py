import os
import glob
prefix = 'slurm_14105804_'
pattern = os.path.join('', f"{prefix}*")
files_to_delete = glob.glob(pattern)

# Loop through the list of files and delete them
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
