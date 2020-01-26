"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

from pytorch_pretrained_bert import BertForNextSentencePrediction, BertConfig

from metrics import f1_score
from metrics import classification_report

from data_loader import DataLoader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='..\\NER_data\\MSRA', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments\\base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--restore_file', default='best', help="name of the file in `model_dir` containing weights to load")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def evaluate(encoder, decoder, data_iterator, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    # idx2tag = params.idx2tag

    true_answers = []
    pred_answers = []

    # a running average object for loss
    loss_avg = utils.RunningAverage()

    for _ in range(params.eval_steps):
        # fetch the next evaluation batch
        batch_data, batch_answers, batch_max_len_target= next(data_iterator)
        batch_masks = batch_data.gt(0)
        # compute model output and loss
        encoder_outputs, encoder_hidden = encoder(batch_data, token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:decoder.n_layers]


        for t in range(batch_max_len_target):
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

        if params.n_gpu > 1 and params.multi_gpu:
            loss = loss.mean()
        loss_avg.update(loss.item())
    return loss
        
    #     batch_output = model(batch_data, token_type_ids=None, attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)
        
    #     batch_output = batch_output.detach().cpu().numpy()
    #     batch_answers = batch_answers.to('cpu').numpy()

    #     pred_answers.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])
    #     true_answers.extend([idx2tag.get(idx) for indices in batch_answers for idx in indices])
    # assert len(pred_answers) == len(true_answers)

    # # logging loss, f1 and report
    # metrics = {}
    # f1 = f1_score(true_answers, pred_answers)
    # metrics['loss'] = loss_avg()
    # metrics['f1'] = f1
    # metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    # logging.info("- {} metrics: ".format(mark) + metrics_str)

    # if verbose:
    #     report = classification_report(true_answers, pred_answers)
    #     logging.info(report)
    # return metrics


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

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
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_loader = DataLoader(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    # Load data
    test_data = data_loader.load_data('test')

    # Specify the test set size
    params.test_size = test_data['size']
    params.eval_steps = params.test_size // params.batch_size
    test_data_iterator = data_loader.data_iterator(test_data, shuffle=False)

    logging.info("- done.")

    # Define the model
    config_path = os.path.join(args.bert_model_dir, 'bert_config.json')
    config = BertConfig.from_json_file(config_path)
    model = BertForTokenClassification(config, num_labels=len(params.tag2idx))

    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("Starting evaluation...")
    test_metrics = evaluate(model, test_data_iterator, params, mark='Test', verbose=True)

