import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, 'data.csv')

df = pd.read_csv(csv_file)
'''
df['Review'] = df['Review'].str.strip()  # Remove leading and trailing whitespace
df['Review'].to_csv(current_dir+'/review.txt', index=False, header=False) 
'''
average_length = df['Review'].str.len().mean()
print("Average length of strings in the 'Review' column:", average_length)

max_length = df['Review'].str.len().max()
min_length = df['Review'].str.len().min()
print("Maximum length of strings in the 'Review' column:", max_length)
print("Minimum length of strings in the 'Review' column:", min_length)
max_review = df['Review'][df['Review'].str.len() == max_length].values[0]
min_review = df['Review'][df['Review'].str.len() == min_length].values[0]

print("Content of the cell with maximum length:", max_review)
print("Content of the cell with minimum length:", min_review)