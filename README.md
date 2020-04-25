# DemoNLP

## Requirements
Recommend using [Anaconda3.7](https://docs.anaconda.com/anaconda/install/)

Other API needed:
- [PyTorch](https://pytorch.org/):  
    For Windows and Linux
    ```python
    #cpu ver
    conda install pytorch torchvision cpuonly -c pytorch
    #gpu ver, choose your cuda version as 10.1 as an example.
    conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
    ```
    For MacOS
    ```
    conda install pytorch torchvision -c pytorch
    ```
- Jieba: 
    ```
    pip install jieba
    ```
- re
- unicodedata
- tqdm

## Projects:
- [FreeChat](https://github.com/shinoyuki222/PyTorch_NLP/tree/master/FreeChat)
- [NER](https://github.com/shinoyuki222/PyTorch_NLP/tree/master/NER)
