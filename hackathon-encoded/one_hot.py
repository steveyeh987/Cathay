# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 15:58:36 2017

@author: dadayeh
"""
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np
import pandas as pd

cctxn = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cctxn.csv')
cctxn = cctxn.astype(str)
#enc_onehot = OneHotEncoder()
#train_cat_data = enc_onehot.fit_transform(cctxn)
#train = train_cat_data.toarray()
train = pd.get_dummies(cctxn)



cctxn = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cctxn.csv', index_col=0)
atm = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_atm.csv', index_col=0)
mybank = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_mybank.csv', index_col=0)
cti = pd.read_csv('I:/b02611023/hackntu_x_cathay_2017-master/hackntu_x_cathay_2017-master/hackathon-encoded/final_cti.csv', index_col=0)

frames = [cctxn, atm, mybank, cti]
result = pd.concat(frames)
re = result.drop(['actor_id','action_time'], 1)
i = -1
id = []
for index, x in enumerate(re.duplicated()):
    if x == False:
        i += 1
    id.append(str(i))
result['id'] = id
    
sort = result.sort(['actor_id','action_time'])
seq = []
for i in set(result['actor_id']):
    a_id = result[result['actor_id']==i]
    seq.append(a_id.drop(['actor_id','action_time'], 1))
#sort = sort.drop(['actor_id','action_time'], 1)
le = LabelEncoder()
re = sort.astype(str).apply(le.fit_transform)
enc_onehot = OneHotEncoder()
train_cat_data = enc_onehot.fit_transform(re)
train = train_cat_data.toarray()
one_hot_targets = np.eye(len(re))
#onehot = pd.get_dummies(result.astype(str))


import json
from collections import Counter, OrderedDict
import numpy as np
import random
import math

def LearnVocabFromTrainFile():
        
    # 開啟唐詩語料庫

    #f = open("poem.txt")
 
    # 統計唐詩語料庫中每個字出現的頻率

    vcount = Counter(result['id'])
            
    # 僅保留出現次數大於五的字，並按照出現次數排序

    vcount_list = sorted(filter(lambda x: x[1] >= 1, vcount.items())
                         , reverse=True, key=lambda x: x[1])
                         
    # 建立字典，將每個字給一個id ，字為 key, id 為 value

    vocab_dict = OrderedDict(map(lambda x: (x[1][0], x[0]), enumerate(vcount_list)))
    
    # 建立詞頻統計用的字典，給定某字，可查到其出現頻率

    vocab_freq_dict = OrderedDict(map(lambda x: (x[0], x[1]), vcount_list))
    return vocab_dict, vocab_freq_dict
    
vocab_dict, vocab_freq_dict =  LearnVocabFromTrainFile()

def InitUnigramTable(vocab_freq_dict):
    table_freq_list = map(lambda x: (x[0], int(x[1][1] ** 0.75)), enumerate(vocab_freq_dict.items()))
    table_size = sum([x[1] for x in table_freq_list])
    table = np.zeros(table_size).astype(int)
    offset = 0
    for item in table_freq_list:
        table[offset:offset + item[1]] = item[0]
        offset += item[1]

    return table
    
table = InitUnigramTable(vocab_freq_dict)

def train(vocab_dict, vocab_freq_dict, table):
        
    total_words = sum([x[1] for x in vocab_freq_dict.items()])
    vocab_size = len(vocab_dict)

    # 參數設定

    layer1_size = 30 # hidden layer 的大小，即向量大小

    window = 2 # 上下文寬度的上限

    alpha_init = 0.025 # learning rate

    sample = 0.001 # 用來隨機丟棄高頻字用

    negative = 10 # negative sampling 的數量

    ite = 2 # iteration 次數

    
    # Weights 初始化

    # syn0 : input layer 到 hidden layer 之間的 weights ，用隨機值初始化

    # syn1 : hidden layer 到 output layer 之間的 weights ，用0初始化

    syn0 = (0.5 - np.random.rand(vocab_size, layer1_size)) / layer1_size 
    syn1 = np.zeros((layer1_size, vocab_size))
    
    # 印出進度用

    train_words = 0 # 總共訓練了幾個字

    p_count = 0
    avg_err = 0.
    err_count = 0
    
    for local_iter in range(ite):
        print("local_iter", local_iter)
        f = seq
        for actor in f:
            
            #用來暫存要訓練的字，一次訓練一個句子

            sen = []
            
            # 取出要被訓練的字

            for word_raw in actor['id']:
                last_word = vocab_dict.get(word_raw, -1)
                
                # 丟棄字典中沒有的字（頻率太低）

                if last_word == -1:
                    continue
                cn = vocab_freq_dict.get(word_raw)
                ran = (math.sqrt(cn / float(sample * total_words + 1))) * (sample * total_words) / cn
                
                # 根據字的頻率，隨機丟棄，頻率越高的字，越有機會被丟棄

                if ran < random.random():
                    continue
                train_words += 1
                
                # 將要被訓練的字加到 sen

                sen.append(last_word)
                
            # 根據訓練過的字數，調整 learning rate

            alpha = alpha_init * (1 - train_words / float(ite * total_words + 1))
            if alpha < alpha_init * 0.0001:
                alpha = alpha_init * 0.0001
                
            # 逐一訓練 sen 中的字

            for a, word in enumerate(sen):
            
                    # 隨機調整 window 大小

                b = random.randint(1, window)
                for c in range(a - b, a + b + 1):
                    
                    # input 為 window 範圍中，上下文的某一字

                    if c < 0 or c == a or c >= len(sen):
                        continue
                    last_word = sen[c]
                                        
                    # h_err 暫存 hidden layer 的 error 用

                    h_err = np.zeros((layer1_size))
                    
                    # 進行 negative sampling

                    for negcount in range(negative):
                    
                            # positive example，從 sen 中取得，模型要輸出 1

                        if negcount == 0:
                            target_word = word
                            label = 1
                        
                        # negative example，從 table 中抽樣，模型要輸出 0 

                        else:
                            while True:
                                target_word = table[random.randint(0, len(table) - 1)]
                                if target_word not in sen:
                                    break
                            label = 0
                        
                        # 模型預測結果

                        o_pred = 1 / (1 + np.exp(- np.dot(syn0[last_word, :], syn1[:, target_word])))
                        
                        # 預測結果和標準答案的差距

                        o_err = o_pred - label
                        
                        # backward propagation

                        # 此部分請參照 word2vec part2 的公式推導結果

                        
                        # 1.將 error 傳遞到 hidden layer                        

                        h_err += o_err * syn1[:, target_word]
                        
                        # 2.更新 syn1

                        syn1[:, target_word] -= alpha * o_err * syn0[last_word]
                        avg_err += abs(o_err)
                        err_count += 1
                    
                    # 3.更新 syn0

                    syn0[last_word, :] -= alpha * h_err
                    
                    # 印出目前結果

                    p_count += 1
                    if p_count % 10000 == 0:
                        print("Iter: %s, Alpha %s, Train Words %s, Average Error: %s" \
                              % (local_iter, alpha, 100 * train_words, avg_err / float(err_count)))
                        avg_err = 0.
                        err_count == 0.
                        
        # 每一個 iteration 儲存一次訓練完的模型

        model_name = "w2v_model_blog_%s.json" % (local_iter)
        print("save model: %s" % (model_name))
        fm = open(model_name, "w")
        fm.write(json.dumps(syn0.tolist(), indent=4))
        fm.close()

train(vocab_dict, vocab_freq_dict, table)






























def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df
#ooo = onehot(result, list(result))

#from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
#tsne = TSNE(n_components=2)
#X_tsne = tsne.fit_transform(onehot[0:2,:])
#
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
#plt.show()

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=2)
X_pca = pca.fit_transform(onehot)  
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.show()




import pandas as pd
#import numpy as np
from sklearn.feature_extraction import DictVectorizer
 
def encode_onehot(df, cols):
    """
    One-hot encoding is applied to columns specified in a pandas DataFrame.
    
    Modified from: https://gist.github.com/kljensen/5452382
    
    Details:
    
    http://en.wikipedia.org/wiki/One-hot
    http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(orient='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df


def main():
  np.random.seed(42)
  df = pd.DataFrame(np.random.randn(25, 3), columns=['a', 'b', 'c'])

  # Make some random categorical columns
  df['e'] = [np.random.choice(('Chicago', 'Boston', 'New York')) for i in range(df.shape[0])]
  df['f'] = [np.random.choice(('Chrome', 'Firefox', 'Opera', "Safari")) for i in range(df.shape[0])]

  # Vectorize the categorical columns: e & f
  df = encode_onehot(df, cols=['e', 'f'])
  print(df.head())

if __name__ == '__main__':
    main()




