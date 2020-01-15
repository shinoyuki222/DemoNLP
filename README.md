# Pytorch NLP

## Requirements
Recommend using [Anaconda3.7](https://docs.anaconda.com/anaconda/install/), other API needed:
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
