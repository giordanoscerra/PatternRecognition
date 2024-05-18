import subprocess
from itertools import product
import multiprocessing

dataset = "review.txt"
n_epochs_values = [50, 100, 500, 1000, 20000]
hidden_size_values = [150, 50, 100]
n_layers_values = [5,4,3,2]
model_values = ["lstm", "gru"]
learning_rate_values = [0.01]
chunk_len_values = [100, 200]
batch_size_values = [100]
regularize_values = [0, 0.2, 0.5]


def run_training(n_epochs, hidden_size, n_layers, model, learning_rate, chunk_len, batch_size, regularize):
    args = ["--n_epochs", str(n_epochs), 
            "--hidden_size", str(hidden_size), 
            "--n_layers", str(n_layers), 
            "--model", model, 
            "--learning_rate", str(learning_rate), 
            "--chunk_len", str(chunk_len), 
            "--batch_size", str(batch_size), 
            "--regularize", str(regularize), 
            dataset]
    command = ["python", "train.py"] + args
    subprocess.run(command)

if __name__ == "__main__":
    processes = []
    max_processes = 4
    for n_epochs, hidden_size, n_layers, model, learning_rate, chunk_len, batch_size, regularize in product(n_epochs_values, hidden_size_values, n_layers_values, model_values, learning_rate_values, chunk_len_values, batch_size_values, regularize_values):
        p = multiprocessing.Process(target=run_training, args=(n_epochs, hidden_size, n_layers, model, learning_rate, chunk_len, batch_size, regularize))
        p.start()
        processes.append(p)
        
        if len(processes) >= max_processes:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()


