import os
import subprocess

dataset = 'totti'
folder = dataset+'_models'
output_file = dataset+'_output.txt'
i = 0

temperature = "1.15"
start_token = "Totti"

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