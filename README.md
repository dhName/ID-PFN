# ID-PFN
**I**terative **D**enoising Method with **P**attern **F**usion **N**etwork

# Requirements
pip install -r requirements.txt

* python==3.6
* tensorflow==1.13.1
* numpy==1.16.4
* tqdm==4.40.2
* nltk==3.4.5
* matplotlib==3.0.3
* django==3.0.3


# Framework
####  Trainning (**I**terative **D**enoising Method)

![image](https://user-images.githubusercontent.com/42259606/112608089-df51a900-8e54-11eb-9657-e47f162dc316.png)

#### Neural Network（**P**attern **F**usion **N**etwork）

![image](https://user-images.githubusercontent.com/42259606/112608111-e5e02080-8e54-11eb-8475-bfd23360b722.png)


# metric
#### precison、recall、F1-score

| 模型 | precision | recall | F1-score |
| :-----| ----: | :----: | :----: |
| CNN+ONE | 32.95 | 65.42 | 43.83 |
| CNN+ATT | 34.92 | 64.24 | 45.25 |
| PCNN+ONE | 33.01 | 70.12 | 44.89 |
| PCNN+ATT | 34.84 | 73.06 | 47.18 |
| CNN+RL | **45.23** | 70.78 | 55.19 |
| LSTM | 34.92 | 66.39 | 45.77 |
| LSTM+ATT | 35.33 | 65.27 | 45.84 |
| ID+PFN | 44.75 | **80.61** | **57.55** |

#### P-R curve

![image](https://user-images.githubusercontent.com/42259606/111864113-18dc6d00-899a-11eb-8c6b-da3866e23c58.png)


# Systemn Interface
We implement the ID+PFN method to relation extraction system by Django which is a popular web project. 

The first button is index page, the second is training parameters setting, the third is pattern which is mined by unsupervided method, the fourth is training model metrices, and the fifth is relation extraction prediction.

![image](https://user-images.githubusercontent.com/42259606/113532232-7230d700-95fd-11eb-8016-8af738503c37.png)


# Reference
1. Neural Relation Extraction with Selective Attention over Instances. Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, Maosong Sun. ACL2016. [paper](http://www.aclweb.org/anthology/P16-1200)

2. Reinforcement Learning for Relation Classification from Noisy Data. Jun Feng, Minlie Huang, Li Zhao, Yang Yang, Xiaoyan Zhu. AAAI2018. [paper](https://tianjun.me/static/essay_resources/RelationExtraction/Paper/AAAI2018Denoising.pdf)

3. ...
