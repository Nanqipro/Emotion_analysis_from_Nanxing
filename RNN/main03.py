# -*- coding: utf-8 -*- 
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, SimpleRNN
from jieba import lcut
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, accuracy_score, f1_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard

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

X_train, X_t, train_y, v_y = train_test_split(data, label, test_size=0.4, random_state=42)
X_val, X_test, val_y, test_y = train_test_split(X_t, v_y, test_size=0.5, random_state=42)

## 标签编码
le = LabelEncoder()
train_y = le.fit_transform(train_y).reshape(-1, 1)
val_y = le.transform(val_y).reshape(-1, 1)
test_y = le.transform(test_y).reshape(-1, 1)

## One-Hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y).toarray()
val_y = ohe.transform(val_y).toarray()
test_y = ohe.transform(test_y).toarray()

## Tokenizer
max_words = 5000
max_len = 600
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(data)

train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

## RNN模型
inputs = Input(name='inputs', shape=[max_len])
layer = Embedding(max_words + 1, 128, input_length=max_len)(inputs)
layer = SimpleRNN(128)(layer)
layer = Dense(128, activation="relu", name="FC1")(layer)
layer = Dropout(0.5)(layer)
layer = Dense(2, activation="softmax", name="FC2")(layer)
model = Model(inputs=inputs, outputs=layer)
model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["accuracy"])

# 模型训练，并记录训练过程中的 loss 和 accuracy
history = model.fit(train_seq_mat, train_y, batch_size=128, epochs=100, validation_data=(val_seq_mat, val_y),
                    callbacks=[TensorBoard(log_dir='./log')])

# 保存并加载模型
model.save('./model/RNN.h5')
del model
model = load_model('./model/RNN.h5')

## 对验证集进行预测
test_pre = model.predict(test_seq_mat)
pred = np.argmax(test_pre, axis=1)
real = np.argmax(test_y, axis=1)

# 计算各项评估指标
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='micro')
recall = recall_score(real, pred, average='micro')
f1 = f1_score(real, pred, average='micro')

# 打印评估结果
print(f"Test: acc: {acc:.4f}   precision: {precision:.4f}   recall: {recall:.4f}   f1: {f1:.4f}")

# 分类报告
print("\nClassification Report:")
print(classification_report(real, pred))

# 绘制混淆矩阵
cv_conf = confusion_matrix(real, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=['negative', 'active'])
disp.plot(cmap="Blues", values_format='')
plt.savefig("./results/ConfusionMatrix.png", dpi=400)

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(real, test_pre[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.savefig("./results/ROC_Curve.png", dpi=400)

# 比较实际值与预测值
plt.figure(figsize=(10, 7))
plt.plot(real, label='True Values', color='blue', alpha=0.6)
plt.plot(pred, label='Predicted Values', color='red', alpha=0.6)
plt.title("Comparison of Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel("Class Labels")
plt.legend()
plt.savefig("./results/Actual_vs_Predicted.png", dpi=400)

# 绘制loss曲线和准确率曲线
# Loss曲线
plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./results/loss.png', dpi=400)

# 准确率曲线
plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('./results/Accuracy_Curve.png', dpi=400)
