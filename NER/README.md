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
#### To train the model
    python train.py
