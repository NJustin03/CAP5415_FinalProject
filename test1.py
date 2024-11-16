import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('cars_train_labels.csv')

# Specify the column you want to subtract from (replace 'column_name' with the actual column name)
df['class'] = df['class'].apply(lambda x: x - 1)

# Save the modified DataFrame back to a CSV file (optional)
df.to_csv('modified_file.csv', index=False)

# Print the modified DataFrame to see the changes
print(df)
