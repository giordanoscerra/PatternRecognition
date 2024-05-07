import subprocess
from itertools import product
import os

os.chdir("./3")

arguments = []
n_epochs_values = [1, 500, 1000]
hidden_size_values = [50, 100, 150]
n_layers_values = [2, 3, 4]
model_values = ["gru", "lstm"]
learning_rate_values = [0.1, 0.01, 0.001]
chunk_len_values = [200, 100, 50]
batch_size_values = [100, 50, 25]

for n_epochs, hidden_size, n_layers, model, learning_rate, chunk_len, batch_size in product(n_epochs_values, hidden_size_values, n_layers_values, model_values, learning_rate_values, chunk_len_values, batch_size_values):
    args = ["--n_epochs", str(n_epochs), "--hidden_size", str(hidden_size), "--n_layers", str(n_layers), "--model", model, "--learning_rate", str(learning_rate), "--chunk_len", str(chunk_len), "--batch_size", str(batch_size), "review.txt"]
    command = ["python", "train.py"] + args
    subprocess.run(command)


