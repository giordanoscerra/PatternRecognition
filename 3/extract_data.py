import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_dir, 'data.csv')

df = pd.read_csv(csv_file)

df['Review'] = df['Review'].str.strip()  # Remove leading and trailing whitespace
df['Review'].to_csv(current_dir+'/review.txt', index=False, header=False) 

