#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from tqdm import tqdm

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=100)
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
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / args.chunk_len

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
all_losses = []
loss_avg = 0

try:
    save_filename = 'M=' + model_name + '_E=' + str(args.n_epochs) + '_HS=' + str(args.hidden_size) + '_HL=' + str(args.n_layers) + '_LR=' + str(args.learning_rate) + '_CL=' + str(args.chunk_len) + '_BS=' + str(args.batch_size) + '.pt'
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        loss_avg += loss

        if epoch % args.print_every == 0:
            print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / args.n_epochs * 100, loss))
            print('\n', '----------', '\n', generate(decoder, 'The', 100, cuda=args.cuda), '\n', '----------', '\n')

    #print("generating text with", save_filename)
    #print('\n', '----------', '\n', generate(decoder, 'The', 100, cuda=args.cuda), '\n', '----------', '\n')
    #print("Saving...")
    save_folder = os.path.splitext(args.filename)[0]+'_models'
    save(save_filename, save_folder)

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

