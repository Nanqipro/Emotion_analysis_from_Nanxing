# -*- coding: utf-8 -*-
# V1
# 加入评估方法：增加了准确率、召回率、精确度、F1得分、混淆矩阵、ROC曲线等评估指标，并进行了可视化。
# 交叉验证：使用了Stratified K-Fold交叉验证来评估模型性能。
# 探索性数据分析（EDA）：添加了数据分布、文本长度分布等初步的EDA步骤。
# 实际值与预测值对比图：绘制了实际值与预测值的对比图。
# 结果保存：所有生成的图表均保存到指定的results文件夹中。

import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
import os
import pandas as pd
import matplotlib.ticker as mtick

# 确保保存结果的文件夹存在
if not os.path.exists('results'):
    os.makedirs('results')

# 超参数设置
data_path =  './data/data.txt'              # 数据集
vocab_path = './data/vocab.pkl'             # 词表
save_path = './saved_dict/lstm.ckpt'        # 模型训练结果
embedding_pretrained = \
    torch.tensor(
    np.load(
        './data/embedding_Tencent.npz')
    ["embeddings"].astype('float32'))
                                            # 预训练词向量
embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.5                               # 随机丢弃
num_classes = 2                             # 类别数
num_epochs = 200                            # epoch数
batch_size = 128                            # mini-batch大小
pad_size = 50                               # 每句话处理成的长度(短填长切)
learning_rate = 1e-3                        # 学习率
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数
MAX_VOCAB_SIZE = 10000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号

def get_data():
    tokenizer = lambda x: [y for y in x]  # 字级别
    vocab = pkl.load(open(vocab_path, 'rb'))
    # print('tokenizer',tokenizer)
    print('vocab',vocab)
    print(f"Vocab size: {len(vocab)}")

    train, dev, test = load_dataset(data_path, pad_size, tokenizer, vocab)
    return vocab, train, dev, test

def load_dataset(path, pad_size, tokenizer, vocab):
    '''
    将路径文本文件分词并转为三元组返回
    :param path: 文件路径
    :param pad_size: 每个序列的大小
    :param tokenizer: 转为字级别
    :param vocab: 词向量模型
    :return: 二元组，含有字ID，标签
    '''
    contents = []
    n=0
    with open(path, 'r', encoding='gbk') as f:
        # tqdm可以看进度条
        for line in tqdm(f):
            # 默认删除字符串line中的空格、’\n’、't’等。
            lin = line.strip()
            if not lin:
                continue
            # print(lin)
            label,content = lin.split('	####	')
            # word_line存储每个字的id
            words_line = []
            # 分割器，分词每个字
            token = tokenizer(content)
            # print(token)
            # 字的长度
            seq_len = len(token)
            if pad_size:
                # 如果字长度小于指定长度，则填充，否则截断
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # 将每个字映射为ID
            # 如果在词表vocab中有word这个单词，那么就取出它的id；
            # 如果没有，就去除UNK（未知词）对应的id，其中UNK表示所有的未知词（out of vocab）都对应该id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            n+=1
            contents.append((words_line, int(label)))

    # 数据集拆分为train, dev, test
    train, X_t = train_test_split(contents, test_size=0.4, random_state=42, stratify=[x[1] for x in contents])
    dev, test = train_test_split(X_t, test_size=0.5, random_state=42, stratify=[x[1] for x in X_t])
    return train, dev, test

# get_data()

class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self,index):
        self.text = self.x[index]
        self.label = self.y[index]
        return self.text, self.label
    def __len__(self):
        return len(self.x)

# 以上是数据预处理的部分

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 定义LSTM模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 使用预训练的词向量模型，freeze=False 表示允许参数在训练中更新
        # 在NLP任务中，当我们搭建网络时，第一层往往是嵌入层，对于嵌入层有两种方式初始化embedding向量，
        # 一种是直接随机初始化，另一种是使用预训练好的词向量初始化。
        self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # bidirectional=True表示使用的是双向LSTM
        self.lstm = nn.LSTM(embed, hidden_size, num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        # 因为是双向LSTM，所以层数为config.hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out = self.embedding(x)
        # lstm 的input为[batchsize, max_length, embedding_size]，输出表示为 output,(h_n,c_n),
        # 保存了每个时间步的输出，如果想要获取最后一个时间步的输出，则可以这么获取：output_last = output[:,-1,:]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# 权重初始化，默认xavier
# xavier和kaiming是两种初始化参数的方法
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def plot_acc(train_acc, val_acc, fold):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(1, len(train_acc)+1))
    plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='Train Acc')
    plt.plot(x, val_acc, alpha=0.9, linewidth=2, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch for Fold {fold}')
    plt.legend(loc='best')
    plt.savefig(f'results/acc_fold_{fold}.png', dpi=400)
    plt.close()

def plot_loss(train_loss, val_loss, fold):
    sns.set(style='darkgrid')
    plt.figure(figsize=(10, 7))
    x = list(range(1, len(train_loss)+1))
    plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='Train Loss')
    plt.plot(x, val_loss, alpha=0.9, linewidth=2, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss per Epoch for Fold {fold}')
    plt.legend(loc='best')
    plt.savefig(f'results/loss_fold_{fold}.png', dpi=400)
    plt.close()

# 定义训练的过程
def train(model, train_loader, val_loader, fold):
    '''
    训练模型
    :param model: 模型
    :param train_loader: 训练集DataLoader
    :param val_loader: 验证集DataLoader
    :param fold: 当前fold编号
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Start Training Fold {fold}...\n")
    plot_train_acc = []
    plot_val_acc = []
    plot_train_loss = []
    plot_val_loss = []

    for epoch in range(num_epochs):
        # 1，训练循环----------------------------------------------------------------
        model.train()
        step = 0
        train_loss_total = 0
        train_acc_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            train_loss_total += loss.item()
            preds = torch.max(outputs.data, 1)[1].cpu().numpy()
            # train_acc_total += accuracy_score(labels, preds)
            train_acc_total += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())  # 转为 numpy 数组


        train_acc = train_acc_total / step
        train_loss = train_loss_total / step
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)

        # 2，验证集验证----------------------------------------------------------------
        val_acc, val_loss = evaluate(model, val_loader, loss_function)
        plot_val_acc.append(val_acc)
        plot_val_loss.append(val_loss)

        # 保存最好的模型
        if val_loss < dev_best_loss:
            dev_best_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Fold {fold} Epoch {epoch+1}: "
              f"Train Loss={train_loss:.3f}, Train Acc={train_acc:.2%}, "
              f"Val Loss={val_loss:.3f}, Val Acc={val_acc:.2%}")

    # 绘制训练过程中的Acc和Loss
    plot_acc(plot_train_acc, plot_val_acc, fold)
    plot_loss(plot_train_loss, plot_val_loss, fold)

def evaluate(model, data_loader, loss_function):
    '''
    模型评估
    :param model: 模型
    :param data_loader: 数据加载器
    :param loss_function: 损失函数
    :return: 准确率和损失
    '''
    model.eval()
    loss_total = 0
    correct_total = 0
    with torch.no_grad():
        for texts, labels in data_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            preds = torch.max(outputs.data, 1)[1]
            correct_total += (preds == labels).sum().item()
    acc = correct_total / len(data_loader.dataset)
    loss_avg = loss_total / len(data_loader)
    return acc, loss_avg

def result_test(real, pred, probabilities):
    # 计算混淆矩阵
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='binary')
    recall = recall_score(real, pred, average='binary')
    f1 = f1_score(real, pred, average='binary')
    patten = 'test:  acc: %.4f   precision: %.4f   recall: %.4f   f1: %.4f'
    print(patten % (acc, precision, recall, f1,))
    
    # 混淆矩阵可视化
    labels11 = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig("results/ConfusionMatrix.png", dpi=400)
    plt.close()
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(real, probabilities[:,1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig("results/ROC_Curve.png", dpi=400)
    plt.close()

    # 实际值与预测值对比图
    plt.figure(figsize=(10,6))
    plt.scatter(range(len(real)), real, label='Actual', alpha=0.6)
    plt.scatter(range(len(pred)), pred, label='Predicted', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('Actual vs Predicted Labels')
    plt.legend()
    plt.savefig("results/Actual_vs_Predicted.png", dpi=400)
    plt.close()

def plot_eda(data):
    '''
    进行初步的探索性数据分析（EDA）
    :param data: 数据集
    '''
    df = pd.DataFrame(data, columns=['text', 'label'])
    # 文本长度分布
    df['length'] = df['text'].apply(len)
    plt.figure(figsize=(10,6))
    sns.histplot(df['length'], bins=50, kde=True)
    plt.title('Text Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.savefig("results/text_length_distribution.png", dpi=400)
    plt.close()
    
    # 标签分布
    plt.figure(figsize=(6,6))
    sns.countplot(x='label', data=df)
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig("results/label_distribution.png", dpi=400)
    plt.close()
    
    # 可选：词频分析
    # 由于是字级别，可以绘制前20个常见字
    from collections import Counter
    all_words = [word for text, label in data for word in text]
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(20)
    words, counts = zip(*most_common)
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(words), y=list(counts))
    plt.title('Top 20 Most Common Characters')
    plt.xlabel('Character')
    plt.ylabel('Frequency')
    plt.savefig("results/top20_characters.png", dpi=400)
    plt.close()

def plot_metrics(y_true, y_pred, probabilities):
    '''
    绘制各种评估指标的图表
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param probabilities: 预测概率
    '''
    # 混淆矩阵和ROC曲线已在result_test中绘制
    # 这里可以添加更多的可视化，如Precision-Recall曲线等
    pass  # 已在result_test中处理

if __name__ == '__main__':
    # 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    # 进行探索性数据分析（EDA）
    print("Performing Exploratory Data Analysis (EDA)...")
    plot_eda(train_data + dev_data + test_data)
    print("EDA plots saved in 'results/' folder.")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 准备交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = [x[0] for x in train_data]
    y = [x[1] for x in train_data]
    X = np.array(X)
    y = np.array(y)

    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"Starting Fold {fold}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 创建DataLoader
        train_dataset = TextDataset(list(zip(X_train, y_train)))
        val_dataset = TextDataset(list(zip(X_val, y_val)))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        model = Model().to(device)
        init_network(model)

        # 训练模型
        train(model, train_loader, val_loader, fold)

        fold += 1

    # 加载最好的模型
    print("Loading the best model...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # 测试集评估
    test_loader = DataLoader(TextDataset(test_data), batch_size=batch_size, shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss()
    test_loss_total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            test_loss_total += loss.item()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.max(outputs.data, 1)[1].cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    test_acc = accuracy_score(all_labels, all_preds)
    test_loss = test_loss_total / len(test_loader)
    print('================' * 8)
    print('Test Loss: {:.3f}      Test Acc: {:.2%}'.format(test_loss, test_acc))

    # 生成并保存评估结果图
    result_test(all_labels, all_preds, all_probs)

    # 绘制实际值和预测值对比图（仅部分样本以避免图像过于密集）
    sample_indices = np.random.choice(len(all_labels), size=1000, replace=False)
    plt.figure(figsize=(15,6))
    plt.plot(all_labels[sample_indices], label='Actual', alpha=0.7)
    plt.plot(all_preds[sample_indices], label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('Actual vs Predicted Labels (Sample)')
    plt.legend(loc='best')
    plt.savefig("results/Actual_vs_Predicted_Sample.png", dpi=400)
    plt.close()

    print("All evaluation metrics and plots have been saved in the 'results/' folder.")



