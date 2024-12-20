## Task 2：Dialogue Act Classification (DAC)

This dir contains the code of **BERT, ERNIE** model for DAC task. It also supports BERT with Cnn/RNN. 

The code is adapted from https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch.

### Requirements

- python 3.7  
- pytorch 1.1  
- tqdm  
- sklearn  
- tensorboardX
- boto3
- requests
- regex

```shell
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
pip install -r requirements.txt
```

### Preparation

download bert model on dir `bert_pretain`, and ERNIE on dir `ERNIE_pretrain`, with three files below

- pytorch_model.bin  
- bert_config.json  
- vocab.txt  

bert_Chinese

- model: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
- vocab: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt  
- alternative: https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese

- model / vocab: http://image.nghuyong.top/ERNIE.zip  
- alternative: https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw

### Training & Inference & Testing

**Note**: The code uses a single script for training, inference, and evaluation, and decoupling the process requires some modifications.

#### BERT

```shell
python run.py --model bert
```

#### BERT with CNN/RNN

```shell
python run.py --model bert_RCNN
```
