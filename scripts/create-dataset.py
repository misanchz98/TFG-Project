from modules import preprocess

# Step 1: Read dataset
df = preprocess.load_csv_data('../dataset/Occupancy_Estimation.csv')

# Step 2: Recode Room_Occupancy_Count column
preprocess.recode_dataset_output(df)

# Step 3: Filter Date and Time columns
preprocess.remove_time_columns(df)

# Step 4: Save modified dataset
preprocess.save_modified_dataset(df)