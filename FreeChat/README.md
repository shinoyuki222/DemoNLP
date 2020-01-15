# Pytorch NLP - FreeChat
[Main repo](https://github.com/shinoyuki222/PyTorch_NLP)


```
└── shinoyuki222/PyTorch_NLP/tree/master/FreeChat
    │
    ├── chatbot_data
    │   └── core_reduced      # simpy chat bot data
    ├── save
    │   └── chatbot_model     # saved model
    │       └──core_reduced   # name depend on corpus
    │           └──2-2_500    # name depend on {encoder_n_layers}-{decoder_n_layers}_{hiddin_size} 
    └── pt_seq2seq_atten_train.py             
```
#### To train the model
    python pt_seq2seq_atten_train.py

#### To load pretrained model and evaluate 

    python pt_seq2seq_atten_train.py -l -cp 4000 -xt

#### Arguments help
    python pt_seq2seq_atten_train.py -h
