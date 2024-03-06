import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # cpu
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random



def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# 读数据

SampleFeature = []
#data = np.loadtxt('E:/20200616_drug-disease/code and data/drug/drug_data/PPMI/drug_net_fused_result_SNF.txt')
with open('E:/20200616_drug-disease/code and data/drug/drug_data/fused_SNF_10network.txt','r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        temp = line.strip().split('\t')
        SampleFeature.append(temp)

SampleFeature = np.array(SampleFeature)
print('SampleFeature',len(SampleFeature))
print('SampleFeature[0]',len(SampleFeature[0]))
x = SampleFeature
x_train = SampleFeature
x_test = SampleFeature

# 改变数据类型
x_train = x_train.astype('float32') / 1.
x_test = x_test.astype('float32') / 1.
print(x_train.shape)
print(x_test.shape)
print(type(x_train[0][0]))
print(x_train.shape)
print(x_test.shape)
print(type(x_train[0][0]))

# 变量
encoding_dim = 768
input_img = Input(shape=(len(SampleFeature[0]),))    # 输入维度
input_img = Input(shape=(len(SampleFeature[0]),))


# 构建autoencoder
from keras import regularizers
encoded_input = Input(shape=(encoding_dim,))
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(10e-7))(input_img)    # 参数10e-7可调节
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(10e-7))(input_img)
decoded = Dense(1519, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs=input_img, outputs=encoded)
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=50, shuffle=True, validation_data=(x_test, x_test))

# 预测
encoded_imgs = encoder.predict(x)
decoded_imgs = decoder.predict(encoded_imgs)
print(len(encoded_imgs))
print(len(encoded_imgs[1]))
print(len(encoded_imgs))
print(len(encoded_imgs[1]))
storFile(encoded_imgs, '768SampleFeature_10network.csv')