#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=25)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--regularize', type=float, default=0.0)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--cuda', action='store_true')
args = argparser.parse_args()

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    train_size = int(0.8 * file_len)  # 80% of the file for training
    train_file = file[:train_size]
    val_file = file[train_size:]

    train_inp = torch.LongTensor(batch_size, chunk_len)
    train_target = torch.LongTensor(batch_size, chunk_len)
    val_inp = torch.LongTensor(batch_size, chunk_len)
    val_target = torch.LongTensor(batch_size, chunk_len)

    for bi in range(batch_size):
        train_start_index = random.randint(0, train_size - chunk_len - 1)
        train_end_index = train_start_index + chunk_len + 1
        train_chunk = train_file[train_start_index:train_end_index]
        train_inp[bi] = char_tensor(train_chunk[:-1])
        train_target[bi] = char_tensor(train_chunk[1:])

        val_start_index = random.randint(0, len(val_file) - chunk_len - 1)
        val_end_index = val_start_index + chunk_len + 1
        val_chunk = val_file[val_start_index:val_end_index]
        val_inp[bi] = char_tensor(val_chunk[:-1])
        val_target[bi] = char_tensor(val_chunk[1:])

    train_inp = Variable(train_inp)
    train_target = Variable(train_target)
    val_inp = Variable(val_inp)
    val_target = Variable(val_target)

    if args.cuda:
        train_inp = train_inp.cuda()
        train_target = train_target.cuda()
        val_inp = val_inp.cuda()
        val_target = val_target.cuda()

    return train_inp, train_target, val_inp, val_target


def train(train_inp, train_target, val_inp, val_target):
        train_hidden = decoder.init_hidden(args.batch_size)
        val_hidden = decoder.init_hidden(args.batch_size)
        if args.cuda:
            train_hidden = train_hidden.cuda()
            val_hidden = val_hidden.cuda()
        decoder.zero_grad()
        train_loss = 0
        val_loss = 0

        for c in range(args.chunk_len):
            train_output, train_hidden = decoder(train_inp[:,c], train_hidden)
            train_loss += criterion(train_output.view(args.batch_size, -1), train_target[:,c])

            val_output, val_hidden = decoder(val_inp[:,c], val_hidden)
            val_loss += criterion(val_output.view(args.batch_size, -1), val_target[:,c])

        train_loss.backward()
        decoder_optimizer.step()

        return train_loss.item() / args.chunk_len, val_loss.item() / args.chunk_len

def save(save_filename, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(decoder, os.path.join(save_folder, save_filename))
    print('Saved as %s' % save_filename, '\n')

# Initialize models and start training

model_name = args.model

if args.regularize > 0:
    #print("Using RegularizedCharRNN")
    decoder = RegularizedCharRNN(
        n_characters,
        args.hidden_size,
        n_characters,
        model=args.model,
        n_layers=args.n_layers,
        dropout_prob=args.regularize
    )
    
    model_name = model_name + '_dropout'
    
else:
    #print("Using CharRNN")
    decoder = CharRNN(
        n_characters,
        args.hidden_size,
        n_characters,
        model=args.model,
        n_layers=args.n_layers,
    )

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
tr_losses = []
vl_losses = []

try:
    save_filename = 'M=' + model_name + '_E=' + str(args.n_epochs) + '_HS=' + str(args.hidden_size) + '_HL=' + str(args.n_layers) + '_LR=' + str(args.learning_rate) + '_CL=' + str(args.chunk_len) + '_BS=' + str(args.batch_size) + '.pt'
    save_folder = os.path.splitext(args.filename)[0]+'_models'
    model_path = os.path.join(save_folder, save_filename)
    if os.path.exists(model_path):
        print("Model already exists, skipping training")
        exit()
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        tr_loss, vl_loss = train(*random_training_set(args.chunk_len, args.batch_size))
        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        if epoch % args.print_every == 0:
            print('\n','TR_loss: ',tr_loss,' VL_loss: ',vl_loss)
        #    loss_avg = 0
        #    print('\n', '----------', '\n', generate(decoder, 'The', 100, cuda=args.cuda), '\n', '----------', '\n')

    #print("generating text with", save_filename)
    #print('\n', '----------', '\n', generate(decoder, 'The', 100, cuda=args.cuda), '\n', '----------', '\n')
    #print("Saving...")
    plt.plot(tr_losses, label='Training Loss')
    plt.plot(vl_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(model_path + '.png')
    plt.show()
    save(save_filename, save_folder)

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

