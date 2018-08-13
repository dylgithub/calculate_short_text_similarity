#encoding=utf-8
import os
import pandas as pd
import numpy as np
from gensim.models import word2vec
import nltk
from sklearn.preprocessing import OneHotEncoder
def get_stop_words():
    with open('data/stoplist.csv','r') as st:
        word=st.read()
        return word.splitlines()
stopword_list=get_stop_words()[:32]
def get_tes_data():
    print('start get test data')
    sim_label=[]
    sen1=[]
    sen2=[]
    filename_list=os.listdir('data/testdata')
    for filename in filename_list:
        test_df=pd.read_csv('data/testdata/%s' % filename,sep='\t',header=None)
        test_df=test_df[test_df[0].notnull()]
        sim_label.extend(list(test_df[0]))
        sen1.extend(list(test_df[1]))
        sen2.extend(list(test_df[2]))
    sen1_cut_list=[nltk.word_tokenize(sen) for sen in sen1]
    sen2_cut_list=[nltk.word_tokenize(sen) for sen in sen2]
    test_sen1_list=[]
    test_sen2_list=[]
    for sentence1,sentence2 in zip(sen1_cut_list,sen2_cut_list):
        test_sen1_list.append([i for i in sentence1 if i not in stopword_list])
        test_sen2_list.append([j for j in sentence2 if j not in stopword_list])
    print('final get test data')
    return sim_label,test_sen1_list,test_sen2_list
def get_train_data():
    print('start get train data')
    sim_label=[]
    sen1=[]
    sen2=[]
    filename_list=os.listdir('data/traindata')
    for filename in filename_list:
        with open('data/traindata/%s' % filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                newline=line.strip().split('	')
                if len(newline)==3:
                    sim_label.append(newline[0])
                    sen1.append(newline[1])
                    sen2.append(newline[2])
    sen1_cut_list = [nltk.word_tokenize(sen) for sen in sen1]
    sen2_cut_list = [nltk.word_tokenize(sen) for sen in sen2]
    train_sen1_list = []
    train_sen2_list = []
    for sentence1, sentence2 in zip(sen1_cut_list, sen2_cut_list):
        train_sen1_list.append([i for i in sentence1 if i not in stopword_list])
        train_sen2_list.append([j for j in sentence2 if j not in stopword_list])
    print('final get train data')
    return sim_label, train_sen1_list, train_sen2_list
def get_train_model_data():
    print('start get train_model data')
    _, train_sen1_list, train_sen2_list = get_train_data()
    _, test_sen1_list, test_sen2_list = get_tes_data()
    train_model_data = train_sen1_list
    train_model_data.extend(train_sen2_list)
    train_model_data.extend(test_sen1_list)
    train_model_data.extend(test_sen2_list)
    print('final get train_model data')
    return train_model_data
def train_model():
    train_model_data=get_train_model_data()
    print('开始构建词向量')
    word2vec_model = word2vec.Word2Vec(train_model_data, hs=0, min_count=1, window=2, size=80,sg=0)
    word2vec_model.save('word2vec_model/text_cnn_sim_model')
    print('词向量训练完成')
#获得所有单词的词向量并完成单词向索引的映射
def word_pro(vec_size):
    print('start word_pro')
    word2vec_model=word2vec.Word2Vec.load('word2vec_model/text_cnn_sim_model')
    index_dict={}
    word_vecs=[]
    train_model_data = get_train_model_data()
    #注意下面的这一步，它是为了方便对那些不满足规定句子长度的句子进行补0操作
    word_vecs.append(list(np.zeros((vec_size,))))
    for sen in train_model_data:
        for word in sen:
            if word not in list(index_dict.keys()):
                index_dict[word]=len(word_vecs)
                word_vecs.append(list(word2vec_model[word]))
    index_dict['UNKNOW']=len(word_vecs)
    word_vecs.append(list(np.zeros((vec_size,))))
    #注意此处要转换为float32
    word_vecs=np.array(word_vecs).astype(np.float32)
    print('final word_pro')
    return word_vecs,index_dict
#把分词后的数据转换为索引
def get_index_array(sen_list,sen_length,vector_size):
    print('start get_index_array')
    _, index_dict=word_pro(vector_size)
    sen_index=[]
    for sen in sen_list:
        word_index = []
        for word in sen:
            if word in list(index_dict.keys()):
                word_index.append(index_dict[word])
            else:
                word_index.append(index_dict['UNKNOW'])
        sen_len=len(sen)
        if sen_len<sen_length:
            word_index.extend([0]*(sen_length-sen_len))
        sen_index.append(word_index)
    #索引值，保证代码健壮性，向整型转换
    sen_index=np.array(sen_index).astype(np.int32)
    print('final get_index_array')
    return sen_index
def label_process(label):
    label=list(label)
    label=list(map(float,label))
    label=[int(round(float(la))) for la in label]
    label=np.array(label).reshape((-1,1))
    one_hot=OneHotEncoder()
    label=one_hot.fit_transform(label)
    return label.toarray()
# if __name__ == '__main__':
#     train_model()