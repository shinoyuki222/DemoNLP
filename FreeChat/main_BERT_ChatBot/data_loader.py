"""Data loader"""

import random
import numpy as np
import os
import sys

import torch

from pytorch_pretrained_bert import BertTokenizer

import utils


class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model_dir, do_lower_case=True)

    def load_sentences_answers(self, datafiles, d):
        """Loads sentences and answers from their corresponding files. 
            Maps tokens and answers to their indices and stores them in the provided dict d.
        """
        sentences = []
        answers = []
        with open(datafiles, 'r', encoding="utf-8") as file:
            for line in file:
                segs = line.split('|')
                if (len(segs) <= 1):
                    continue
                sentence = segs[0]
                answer = '[CLS]' + segs[1]
                # replace each token by its index
                tokens = self.tokenizer.tokenize(sentence.strip())
                sentences.append(self.tokenizer.convert_tokens_to_ids(tokens))

                tokens = self.tokenizer.tokenize(answer.strip())
                answers.append(self.tokenizer.convert_tokens_to_ids(tokens))
        # storing sentences and answers in dict d
        d['data'] = sentences
        d['answers'] = answers
        d['size'] = len(sentences)

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with answers for each type in types.
        """
        data = {}

        if data_type in ['train', 'dev', 'test']:
            datafile = os.path.join(self.data_dir, data_type, 'qa_paired.txt')
            self.load_sentences_answers(datafile, data)
        else:
            raise ValueError("data type not in ['train', 'dev', 'test']")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with answers.

        Args:
            data: (dict) contains data which has keys 'data', 'answers' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_answers: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size'] // self.batch_size):
            # fetch sentences and answers
            sentences = [data['data'][idx] for idx in order[i *
                                                            self.batch_size:(i + 1) * self.batch_size]]
            answers = [data['answers'][idx]
                       for idx in order[i * self.batch_size:(i + 1) * self.batch_size]]

            # batch length
            batch_len = len(sentences)
            batch_len_target = len(answers)

            # compute length of longest sentence in batch
            batch_max_len_target = max(len(a) for a in answers)
            batch_max_len = max([len(s) for s in sentences])

            max_len = min(batch_max_len, self.max_len)
            max_len_target = min(batch_max_len_target, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_answers = self.token_pad_idx * np.ones((batch_len_target, max_len_target))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                cur_len_target = len(answers[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                else:
                    batch_data[j] = sentences[j][:max_len]

                if cur_len_target < max_len_target:
                    batch_answers[j][:cur_len_target] = answers[j]
                else:
                    batch_answers[j] = answers[j][:max_len_target]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_answers = torch.tensor(batch_answers, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_answers = batch_data.to(
                self.device), batch_answers.to(self.device)

            yield batch_data, batch_answers, batch_max_len_target


if __name__ == '__main__':
    import argparse
    import logging
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='..\\chatbot_data\\QA',
                        help="Directory containing the dataset")
    parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch',
                        help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--model_dir', default='experiments\\base_model',
                        help="Directory containing params.json")
    parser.add_argument('--seed', type=int, default=2019,
                        help="random seed for initialization")
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
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
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
    data_loader = DataLoader(
        args.data_dir, args.bert_model_dir, params, token_pad_idx=0)

    # Load training data and test data
    train_data = data_loader.load_data('train')
    val_data = data_loader.load_data('dev')
    logging.info("datasets loaded.")

    # Specify the training and validation dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    train_data_iterator = data_loader.data_iterator(train_data, shuffle=False)
