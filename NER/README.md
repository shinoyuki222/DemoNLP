# Pytorch NLP - Name Entity Recognition
[Main repo](https://github.com/shinoyuki222/PyTorch_NLP)

Model based on paper:
[Knowledge Graph Embedding Based Question Answering](http://research.baidu.com/Public/uploads/5c1c9a58317b3.pdf)

```
└── shinoyuki222/PyTorch_NLP/tree/master/NER
        |-- README.md
        |-- build_msra_dataset_tags.py
        |-- NER_data
        |   |-- MSRA
        |   |   |-- msra_test_bio
        |   |   |-- msra_train_bio
        |-- main_BERT
        |   |-- data_loader.py
        |   |-- evaluate.py
        |   |-- metrics.py
        |   |-- train.py
        |   |-- utils.py
        |   |-- bert-base-chinese-pytorch
        |   |   |-- bert_config.json
        |   |   |-- pytorch_model.bin
        |   |   |-- vocab.txt
        |   |-- experiments
        |       |-- base_model
        |           |-- evaluate.log
        |           |-- params.json
        |           |-- train.log
        |-- main_LSTM
            |-- consts.py
            |-- dataloader.py
            |-- metric.py
            |-- model.py
            |-- train.py         
```
### To train the LSTM model
```shell
cd main_LSTM
python train.py
```
### To train the BERT-pretrained model
#### Get BERT model for PyTorch
- Install [pytorch-pretrained-bert](https://pypi.org/project/pytorch-pretrained-bert/):
    + pip install pytorch-pretrained-bert
- Convert the TensorFlow checkpoint to a PyTorch dump by yourself
    + Download the Google's BERT base model for Chinese from **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)** (Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters), and decompress it.

    + Execute the following command,  convert the TensorFlow checkpoint to a PyTorch dump.

       ```shell
       export TF_BERT_BASE_DIR=/path/to/chinese_L-12_H-768_A-12
       export PT_BERT_BASE_DIR=/path/to/NER-BERT-pytorch/bert-base-chinese-pytorch
       
       pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch $TF_BERT_BASE_DIR/bert_model.ckpt $TF_BERT_BASE_DIR/bert_config.json $PT_BERT_BASE_DIR/pytorch_model.bin
       ```

    + Copy the BERT parameters file `bert_config.json` and dictionary file `vocab.txt` to the directory `$PT_BERT_BASE_DIR`.

       ```shell
       cp $TF_BERT_BASE_DIR/bert_config.json $PT_BERT_BASE_DIR/bert_config.json
       cp $TF_BERT_BASE_DIR/vocab.txt $PT_BERT_BASE_DIR/vocab.txt
       ```
####
```shell
cd main_BERT
python train.py
```