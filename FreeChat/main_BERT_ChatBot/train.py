"""Train and evaluate the model"""

import argparse
import random
import logging
import os

import torch
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from pytorch_pretrained_bert import BertModel
from model import LuongAttnDecoderRNN
from data_loader import DataLoader
from evaluate import evaluate
import utils





def train(encoder, decoder, data_iterator, encoder_optimizer, decoder_optimizer, scheduler, params):
    """Train the model on `steps` batches"""
    # set model to training mode
    encoder.train()
    decoder.train()
    scheduler.step()

    # a running average object for loss
    loss_avg = utils.RunningAverage()
    
    # Use tqdm for progress bar
    tqdm_t = trange(params.train_steps)
    for i in tqdm_t:
        # fetch the next training batch
        batch_data, batch_tags = next(data_iterator)
        batch_masks = batch_data.gt(0)

        # compute model output and loss
        encoder_outputs, encoder_hidden = encoder(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < params.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * nTotal)
                n_totals += nTotal





        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu

        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=params.clip_grad)

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    

def train_and_evaluate(encoder, decoder,, train_data, val_data, encoder_optimizer, decoder_optimizer, scheduler, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch."""
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, encoder, encoder_optimizer)
        
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # Compute number of batches in one epoch
        params.train_steps = params.train_size // params.batch_size
        params.val_steps = params.val_size // params.batch_size

        # data iterator for training
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=True)
        # Train for one epoch on training set
        train(encoder, decoder, train_data_iterator, encoder_optimizer, decoder_optimizer, scheduler, params)

        # data iterator for evaluation
        train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
        val_data_iterator = data_loader.data_iterator(val_data, shuffle=False)

        # Evaluate for one epoch on training set and validation set
        params.eval_steps = params.train_steps
        train_metrics = evaluate(encoder, decoder, train_data_iterator, params, mark='Train')
        params.eval_steps = params.val_steps
        val_metrics = evaluate(encoder, decoder, val_data_iterator, params, mark='Dev')
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer.optimizer if args.fp16 else optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model_to_save.state_dict(),
                               'optim_dict': optimizer_to_save.state_dict()},
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping and logging best f1
        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("Best val f1: {:05.2f}".format(best_val_f1))
            break
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='..\\chatbot_data\\QA', help="Directory containing the dataset")
    parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch',
                        help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--model_dir', default='experiments\\base_model', help="Directory containing params.json")
    parser.add_argument('--seed', type=int, default=2019, help="random seed for initialization")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before training")
    parser.add_argument('--multi_gpu', default=False, action='store_true',
                        help="Whether to use multiple GPUs if available")
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")



    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    bert_config_path = os.path.join(args.bert_model_dir, 'bert_config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    bert_config = utils.Params(bert_config_path)
    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("device: {}, n_gpu: {}, 16-bits training: {}".format(params.device, params.n_gpu, args.fp16))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)
    
    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('dev')

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    # Prepare encoder
    # model = BertPreTrainedModel.from_pretrained(args.bert_model_dir, num_labels=len(params.tag2idx)))
    encoder = BertModel.from_pretrained(args.bert_model_dir)
    encoder.to(params.device)




    if args.fp16:
        encoder.half()

    if params.n_gpu > 1 and args.multi_gpu:
        encoder = torch.nn.DataParallel(encoder)

    # Prepare encoder_optimizer
    if params.full_finetuning:
        param_optimizer = list(encoder.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(encoder.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("lease install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        encoder_optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=params.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        scheduler = LambdaLR(encoder_optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))
        if args.loss_scale == 0:
            encoder_optimizer = FP16_Optimizer(encoder_optimizer, dynamic_loss_scale=True)
        else:
            encoder_optimizer = FP16_Optimizer(encoder_optimizer, static_loss_scale=args.loss_scale)
    else:
        encoder_optimizer = Adam(optimizer_grouped_parameters, lr=params.learning_rate)
        scheduler = LambdaLR(encoder_optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch))


    # prepare decoder
    attn_model = 'dot'
    decoder_n_layers = 2
    decoder = LuongAttnDecoderRNN(attn_model, encoder.embeddings, bert_config.hidden_size, bert_config.vocab_size, decoder_n_layers, bert_config.hidden_dropout_prob)
    decoder.to(device)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=params.learning_rate * params.decoder_learning_ratio)

    # If you have cuda, configure cuda to call
    if device == 'cuda':
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # Train and evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(encoder, decoder, train_data, val_data, encoder_optimizer, decoder_optimizer, scheduler, params, args.model_dir, args.restore_file)
