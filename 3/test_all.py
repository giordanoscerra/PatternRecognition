import os
import subprocess

folder = 'review_models'
output_file = 'output.txt'
i = 0
with open(output_file, 'w') as f:
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            i += 1
            path = os.path.join(folder, file)
            command = ["python", "generate.py", path, "-t", "1.2", "-p", "They didn't"]
            f.write(str(i) + ": " + "model " + file + "\n")
            result = subprocess.run(command, capture_output=True, text=True)
            f.write("\t"+ result.stdout + "\n")
            f.flush()