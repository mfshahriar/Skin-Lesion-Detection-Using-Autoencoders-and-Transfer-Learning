import os
import pandas as pd

# Specify the folder path and CSV file path
folder_path = '../dataset/Combined_ISIC2020_BCN_HAM'
csv_file_path = '../dataset/Combined_ISIC2020_BCN_HAM_metadata.csv'

# Load the CSV file
csv_data = pd.read_csv(csv_file_path)

# Assuming file names are in a column named 'file_column'
file_column = 'image_name'  # Change this to the actual column name with file names
csv_files = [f + '.jpg' for f in csv_data[file_column].tolist()]

# Get list of files in the specified folder
folder_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Initialize arrays
files_in_folder_not_in_csv = []
files_in_csv_not_in_folder = []

# Check files in folder against files in CSV
for file in folder_files:
    if file not in csv_files:
        files_in_folder_not_in_csv.append(file)

# Check files in CSV against files in folder
for file in csv_files:
    if file not in folder_files:
        files_in_csv_not_in_folder.append(file)

# Print the results
print("Files in folder not in CSV:", files_in_folder_not_in_csv)
print("Files in CSV not in folder:", files_in_csv_not_in_folder)
