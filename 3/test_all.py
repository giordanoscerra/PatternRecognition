import os
import subprocess

dataset = 'review'
folder = dataset+'_models/grid4 interesting models'
output_file = dataset+'_output.txt'
i = 0

temperature = "1.1"
start_token = "Wow."

with open(output_file, 'w') as f:
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            i += 1
            path = os.path.join(folder, file)
            command = ["python", "generate.py", path, "-t", temperature, "-p", start_token]
            f.write(str(i) + ": " + "model " + file + "\n")
            result = subprocess.run(command, capture_output=True, text=True)
            f.write("\t"+ result.stdout + "\n")
            f.flush()