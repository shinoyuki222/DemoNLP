# Pytorch NLP - FreeChat
[Main repo](https://github.com/shinoyuki222/PyTorch_NLP)

Model based on paper:
[Knowledge Graph Embedding Based Question Answering](http://research.baidu.com/Public/uploads/5c1c9a58317b3.pdf)

```
└── shinoyuki222/PyTorch_NLP/tree/master/NER
    │
    ├── NER_data
    │   └── MSRA                # MSRA corpus
    ├── save
    │   └── NER_model                # saved model
    │       └──MSRA             # named by corpus
    │           └──2_500    # named by {rnn_n_layers}_{hiddin_size} 
    │── consts.py
    ├── dataloader.py     
    │── model.py
    └── train.py            
```
#### To train the model
    python train.py
