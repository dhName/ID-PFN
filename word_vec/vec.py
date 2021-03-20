# -*- coding: utf-8 -*-
# @time : 2020/2/22  19:31

import json
import unicodedata
import codecs
import numpy as np

def pre_train_word_vec():
    '''
    将数据集进行预处理，生成以一句为一行的corpus.txt文件，用于glove训练词向量
    :return:
    '''
    datafile = [r"data/train.json",r"data/test.json",r"data/dev.json"]

    fout = open("corpus.txt","w",encoding="utf8")


    for filepath in datafile:
        with open(filepath,"r",encoding="utf8") as fin:
            for line in fin:
                line = json.loads(line.strip())
                sentence = unicodedata.normalize('NFKD', line['sentText'].strip()).encode('ascii', 'ignore').decode()
                words = sentence.split(' ')
                fout.write(' '.join(words)+"\n")
                fout.flush()
    fout.close()

def get_glove_word_vec():
    '''
    https://github.com/stanfordnlp/GloVe
    myblog: https://www.cnblogs.com/dhName/p/12353339.html
    :return:
    '''
    return

def word_vec_npy():
    '''
    读取通过glove训练的词向量文件vectors.txt,来dump np 文件
    :return:
    '''
    word2id={}
    word_vecs=[]

    word2id['[UNK]']=0
    word_vecs.append([0.0 for temp_i in range(100)])

    word_id_count=1
    with open("vectors.txt","r",encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            raw = line.strip().split()
            word = raw[0]
            vec = list(map(float, raw[1:]))
            word2id[word] =word_id_count
            word_vecs.append(vec)
            word_id_count+=1

    json.dump(word2id, codecs.open("word2id.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    np.save("word_vec.npy", np.array(word_vecs, dtype=np.float32))
    print('word vector handle over!', 'word list size is ', word_id_count)


if __name__ == '__main__':
    pre_train_word_vec()
    word_vec_npy()















