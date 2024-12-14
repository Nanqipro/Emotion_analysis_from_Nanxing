# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from jieba import lcut
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# 设置GPU
# 检查可用的GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # 使用第一个GPU
    tf.config.experimental.set_memory_growth(physical_devices[0], True)  # 防止占用全部显存
else:
    print("没有可用的GPU，使用CPU进行训练")

# 数据处理
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str
def dataParse(text, stop_words):
    label, content = text.split('	####	')
    content = reserve_chinese(content)
    words = lcut(content)
    words = [i for i in words if not i in stop_words]
    return words, int(label)
def getStopWords():
    file = open('./data/stopwords.txt', 'r', encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

def getFormatData():
    file = open('./data/data.txt', 'r', encoding='gbk')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)
    return all_words, all_labels

## 读取数据集
data, label = getFormatData()
print(np.shape(data))

X_train, X_t, train_y, v_y = train_test_split(data, label, test_size=0.4, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t, v_y, test_size=0.5, random_state=42)

## 对数据集的标签数据进行编码
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)
test_y = le.transform(test_y).reshape(-1, 1)
print(train_y.shape)
print(val_y.shape)
print(test_y.shape)

## 对数据集的标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()

## 使用Tokenizer对词组进行编码
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)  ## 使用的最大词语数为5000
tok.fit_on_texts(data)

train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

## 将每个序列调整为相同的长度
train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

## 定义RNN模型
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
layer = SimpleRNN(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.summary()

# 编译模型
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# 模型训练
model_fit = model.fit(train_seq_mat, train_y, batch_size=128, epochs=10,
                      validation_data=(val_seq_mat, val_y),
                      callbacks=[TensorBoard(log_dir='./log')])  ## 当val-loss不再提升时停止训练

# 保存模型
model.save('./model/RNN.h5')
del model

# 导入已经训练好的模型
model = load_model('./model/RNN.h5')

## 对验证集进行预测
test_pre = model.predict(test_seq_mat)
pred = np.argmax(test_pre, axis=1)
real = np.argmax(test_y, axis=1)
cv_conf = confusion_matrix(real, pred)

# 计算评价指标
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')
patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
print(patten % (acc, precision, recall, f1))

# 绘制混淆矩阵
labels11 = ['negative', 'active']
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
disp.plot(cmap="Blues", values_format='')
plt.savefig("ConfusionMatrix.tif", dpi=400)
