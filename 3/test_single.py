import os
import subprocess

dataset = 'review'
folder = dataset+'_models/'
model = 'M=lstm_dropout_E=500_HS=150_HL=3_LR=0.01_CL=100_BS=100.pt'
output_file = model+'_output.txt'

temperature_list = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"]
start_token = "Good! "

with open(output_file, 'w') as f:
    path = os.path.join(folder, model)
    print(path)
    for temperature in temperature_list:
        command = ["python", "generate.py", path, "-t", temperature, "-p", start_token]
        f.write("Temperature: " + str(temperature) + "\n")
        result = subprocess.run(command, capture_output=True, text=True)
        f.write("\t" + result.stdout + "\n")
        f.flush()