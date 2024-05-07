import os
import subprocess

for file in os.listdir('.'):
        if file.endswith('.pt'):
            command = ["python", "generate.py", file, "-t", "0.1"]
            print("Generating text with", file)
            subprocess.run(command)