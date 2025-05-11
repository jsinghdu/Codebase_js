import pandas as pd

import ast  # 
# Load the CSV file
df = pd.read_csv('Data/final_courses_with_paths2.csv')  # Replace with your input file path

# Assuming the column name with the list is 'path'
df['BERT_PRED_PATH'] = df['BERT_PRED_PATH'].apply(lambda x: '.'.join(ast.literal_eval(x)))

# Save the updated DataFrame to a new CSV file
df.to_csv('Data/final_path.csv', index=False)  #