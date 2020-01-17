from typing import Any

import torch
from torch import optim
from consts import *
from model import *
from dataloader import *
from model import AttnRNN
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(input_variable, lengths, target_variable, mask, model):
    # Zero gradients
    model_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Forward pass through model
    output= model(input_variable, lengths)
    loss = NLLLoss(output,target_variable, mask)

    return loss

def train(input_variable, lengths, target_variable, mask, model,embedding,model_optimizer,clip):
    # Zero gradients
    model_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)


    # Forward pass through model
    output= model(input_variable, lengths)
    loss = NLLLoss(output,target_variable,mask)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    model_optimizer.step()

    return loss


def trainIters(model_name, voc, tag, pairs_dct, model, model_optimizer, embedding,rnn_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename=None):
    pairs_train = pairs_dct['train']
    pairs_dev = pairs_dct['dev']
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, tag, [random.choice(pairs_train) for _ in range(batch_size)])
                        for _ in range(n_iteration)]
    dev_batches = [batch2TrainData(voc, tag, [random.choice(pairs_dev) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    print_loss_dev = 0
    best_loss = float('Inf')
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        dev_batch = dev_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        model.train()
        loss = train(input_variable, lengths, target_variable, mask, model,embedding, model_optimizer, clip)
        print_loss += loss

        # Run a evaluate iteration with batch
        input_variable, lengths, target_variable, mask, max_target_len = dev_batch
        model.eval()
        loss_dev = evaluate(input_variable, lengths, target_variable, mask, model)
        print_loss_dev += loss_dev

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print_loss_dev_avg = print_loss_dev / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average train loss: {:.4f}, Average dev loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg, print_loss_dev_avg))
            print_loss = 0
            print_loss_dev = 0

        # Save checkpoint
        if print_loss_dev_avg - best_loss < 0.0:
            print("validation loss {0} is better than {1}, saving checkpoint....".format(print_loss_dev_avg,best_loss))
            best_loss = print_loss_dev_avg
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}_{}'.format(rnn_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'model': model.state_dict(),
                'model_opt': model_optimizer.state_dict(),
                'loss': loss,
                'loss_dev': loss_dev,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


        # # Save checkpoint
        # if (iteration % save_every == 0):
        #     directory = os.path.join(save_dir, model_name, corpus_name,
        #                              '{}_{}'.format(rnn_n_layers, hidden_size))
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     torch.save({
        #         'iteration': iteration,
        #         'model': model.state_dict(),
        #         'model_opt': model_optimizer.state_dict(),
        #         'loss': loss,
        #         'embedding': embedding.state_dict()
        #     }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corpus_name', default='MSRA',help='Store corpus name')
    parser.add_argument('-a', '--attn_model', action='store', dest='attn_model', default='concat',
                        help='Store attention mode dot concat or general')

    parser.add_argument('-hs', '--hidden_size', action="store", dest='hidden_size', default=500, type=int,
                        help='Set hidden_size')
    parser.add_argument('-en', '--rnn_num', action="store", dest='rnn_n_layers', default=2, type=int,
                        help='Set rnn_n_layers')
    parser.add_argument('-dp', '--dropout', action="store", dest='dropout', default=0.1, type=int,
                        help='Set dropout rate')
    parser.add_argument('-b', '--batch_size', action="store", dest='batch_size', default=64, type=int,
                        help='Set batch_size')

    parser.add_argument('-n', '--n_iteration', action="store", dest='n_iteration', default=4000, type=int,
                        help='Set n_iteration')

    parser.add_argument('-s', '--save_every', action="store", dest='save_every', default=500, type=int,
                        help='Set save_every')
    parser.add_argument('-p', '--print_every', action="store", dest='print_every', default=1, type=int,
                        help='Set print_every')

    args = parser.parse_args()

    save_dir = os.path.join("", "save")
    corpus_name = args.corpus_name
    corpus = os.path.join("NER_data", corpus_name)
    datafile_train = os.path.join(corpus, "train")
    datafile_dev = os.path.join(corpus, "val")
    print("corpus_name: {0}, corpus = {1}, datafile_train = {2}".format(corpus_name, corpus, datafile_train))

    # Load/Assemble voc and pairs
    voc, tag, pairs = loadTrainData(corpus_name, datafile_train)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)
    # Trim voc and pairs
    voc, pairs = trimRareWords(voc, pairs, MIN_COUNT)
    # save_static_dict(voc, tag, save_dir)
    # voc,tag = load_static_dict(save_dir,corpus_name)

    # load dev data for evaluate model
    pairs_dev = loadDevData(datafile_dev,tag)


    # Configure models
    model_name = 'NER_model'
    attn_model = args.attn_model
    hidden_size = args.hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    batch_size = args.batch_size
    output_size = tag.num_tags
    print('Building model ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize attentionRNN models
    model = AttnRNN(attn_model,hidden_size, output_size, embedding, rnn_n_layers, dropout)
    # Use appropriate device
    model = model.to(device)
    print('Models built and ready to go!')

    clip = 50.0
    teacher_forcing_ratio = 1.0
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_iteration = args.n_iteration
    print_every = args.print_every
    save_every = args.save_every

    # Ensure dropout layers are in train mode
    model.train()

    # Initialize optimizers
    print('Building optimizers ...')
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # If you have cuda, configure cuda to call
    if device == 'cuda':
        for state in model.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # Run training iterations
    print("Starting Training!")
    pairs_dct = {}
    pairs_dct['train'] = pairs
    pairs_dct['dev'] =pairs_dev
    trainIters(model_name, voc, tag, pairs_dct, model, model_optimizer,embedding, rnn_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name,loadFilename=None)