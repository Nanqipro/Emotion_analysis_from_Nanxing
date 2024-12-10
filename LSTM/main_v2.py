# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold  # Added StratifiedKFold
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc  # Added roc_curve and auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

import pandas as pd  # Added for EDA

import os  # Added to create results directory if not exists

# 超参数设置
data_path = './data/data.txt'              # 数据集
vocab_path = './data/vocab.pkl'            # 词表
save_path = './saved_dict/lstm.ckpt'       # 模型训练结果
embedding_pretrained = \
    torch.tensor(
        np.load(
            './data/embedding_Tencent.npz'
        )["embeddings"].astype('float32'))
                                            # 预训练词向量
embed = embedding_pretrained.size(1)        # 词向量维度
dropout = 0.5                               # 随机丢弃
num_classes = 2                             # 类别数
num_epochs = 30                            # epoch数
batch_size = 128                            # mini-batch大小
pad_size = 50                               # 每句话处理成的长度(短填长切)
learning_rate = 1e-3                        # 学习率
hidden_size = 128                           # lstm隐藏层
num_layers = 2                              # lstm层数
MAX_VOCAB_SIZE = 10000                      # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'                 # 未知字，padding符号

# 确保结果保存文件夹存在
os.makedirs('results', exist_ok=True)
os.makedirs('saved_dict', exist_ok=True)

def get_data():
    tokenizer = lambda x: [y for y in x]  # 字级别
    vocab = pkl.load(open(vocab_path, 'rb'))
    # print('tokenizer',tokenizer)
    print('vocab', vocab)
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
    n = 0
    with open(path, 'r', encoding='gbk') as f:
        # tqdm可以看进度条
        for line in tqdm(f):
            # 默认删除字符串line中的空格、’\n’、't’等。
            lin = line.strip()
            if not lin:
                continue
            # print(lin)
            label, content = lin.split('	####	')
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
            n += 1
            contents.append((words_line, int(label)))

    # 使用stratify确保每个类别在训练、验证和测试集中比例一致
    train, X_t = train_test_split(contents, test_size=0.4, random_state=42, stratify=[x[1] for x in contents])  # Added stratify
    dev, test = train_test_split(X_t, test_size=0.5, random_state=42, stratify=[x[1] for x in X_t])  # Added stratify
    return train, dev, test

class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.x = torch.LongTensor([x[0] for x in data]).to(self.device)
        self.y = torch.LongTensor([x[1] for x in data]).to(self.device)
    def __getitem__(self, index):
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

# 添加数据探索性分析（EDA）函数
def perform_eda(train_data, dev_data, test_data):
    print("Performing Exploratory Data Analysis (EDA)...")
    
    # 标签分布分析
    train_labels = [x[1] for x in train_data]
    dev_labels = [x[1] for x in dev_data]
    test_labels = [x[1] for x in test_data]
    
    label_counts = pd.Series(train_labels).value_counts().rename(index={0: '消极', 1: '积极'})
    plt.figure(figsize=(6,4))
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
    plt.title('Distribution of training set labels')
    plt.xlabel('class')
    plt.ylabel('sample')
    plt.savefig('results/eda_label_distribution.png', dpi=300)
    plt.close()
    
    # 词频分析（前100个高频词）
    all_tokens = [token for sample, _ in train_data for token in sample]
    token_freq = pd.Series(all_tokens).value_counts()[:100]
    plt.figure(figsize=(12,8))
    sns.barplot(x=token_freq.values, y=token_freq.index, palette='viridis')
    plt.title('train 100 pre')
    plt.xlabel('Frequency')
    plt.ylabel('words')
    plt.savefig('results/eda_top_100_words.png', dpi=300)
    plt.close()
    
    print("EDA completed and plots saved to 'results/' folder.")

# 添加绘制实际值与预测值对比图的函数
def plot_actual_vs_predicted(y_true, y_pred, save_path, num_samples=100):
    plt.figure(figsize=(15, 5))
    indices = np.random.choice(len(y_true), num_samples, replace=False)
    plt.plot(y_true[indices], label='Actual', marker='o')
    plt.plot(y_pred[indices], label='Predicted', marker='x')
    plt.xlabel('sample id')
    plt.ylabel('class')
    plt.title('Actual vs. predicted values')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 添加绘制训练准确率和损失的函数
def plot_metrics(train_acc, train_loss, save_acc_path, save_loss_path):
    sns.set(style='darkgrid')
    
    # 绘制训练准确率
    plt.figure(figsize=(10, 7))
    epochs = range(1, len(train_acc) + 1)
    plt.plot(epochs, train_acc, 'bo-', label='accuracy')
    plt.title('change accuracy with training rounds')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(save_acc_path, dpi=300)
    plt.close()
    
    # 绘制训练损失
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, 'ro-', label='loss')
    plt.title('change accuracy with training rounds')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_loss_path, dpi=300)
    plt.close()

# 定义训练的过程，添加更多评估指标和可视化
def train(model, dataloaders):
    '''
    训练模型
    :param model: 模型
    :param dataloaders: 处理后的数据，包含train, dev, test
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    dev_best_loss = float('inf')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Start Training...\n")
    
    plot_train_acc = []
    plot_train_loss = []
    
    for epoch in range(num_epochs):
        # 1，训练循环----------------------------------------------------------------
        model.train()
        step = 0
        train_lossi = 0
        train_acci = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in dataloaders['train']:
            # 训练模式，可以更新参数
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零，防止累加
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            step += 1
            true = labels.data.cpu()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_lossi += loss.item()
            train_acci += metrics.accuracy_score(true, predic)
            
            # 记录所有预测和标签用于计算精确度、召回率和F1得分
            all_preds.extend(predic.numpy())
            all_labels.extend(true.numpy())
        
        # 计算训练集的其他指标
        train_precision = precision_score(all_labels, all_preds, average='binary')
        train_recall = recall_score(all_labels, all_preds, average='binary')
        train_f1 = f1_score(all_labels, all_preds, average='binary')
        
        # 2，验证集验证----------------------------------------------------------------
        dev_acc, dev_loss, dev_precision, dev_recall, dev_f1 = dev_eval(model, dataloaders['dev'], loss_function, Result_test=False)
        
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), save_path)
        
        train_acc = train_acci / step
        train_loss = train_lossi / step
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        
        print("Epoch = {} :  train_loss = {:.3f}, train_acc = {:.2%}, train_precision = {:.2%}, train_recall = {:.2%}, train_f1 = {:.2%}, dev_loss = {:.3f}, dev_acc = {:.2%}, dev_precision = {:.2%}, dev_recall = {:.2%}, dev_f1 = {:.2%}".
              format(epoch+1, train_loss, train_acc, train_precision, train_recall, train_f1, dev_loss, dev_acc, dev_precision, dev_recall, dev_f1))
    
    # 绘制训练准确率和损失
    plot_metrics(plot_train_acc, plot_train_loss, "results/train_accuracy.png", "results/train_loss.png")
    
    # 3，验证循环----------------------------------------------------------------
    model.load_state_dict(torch.load(save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_precision, test_recall, test_f1, y_true, y_pred = dev_eval(model, dataloaders['test'], loss_function, Result_test=True)
    print('================' * 8)
    print('test_loss: {:.3f}      test_acc: {:.2%}'.format(test_loss, test_acc))
    
    # 绘制混淆矩阵和ROC曲线
    labels = ['negative', 'active']
    plot_confusion_matrix(y_true, y_pred, labels, "results/lstm_confusion_matrix.png")
    
    # 获取预测概率用于ROC曲线
    with torch.no_grad():
        y_pred_prob = []
        for texts, labels in dataloaders['test']:
            texts = texts.to(device)
            outputs = model(texts)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred_prob.extend(probs)
    
    # 获取真实标签
    y_true_binary = y_true  # Assuming labels are 0 and 1
    
    plot_roc_curve(y_true_binary, y_pred_prob, "results/lstm_roc_curve.png")
    
    # 绘制实际值与预测值对比图
    plot_actual_vs_predicted(y_true, y_pred, "results/actual_vs_predicted.png")
    
    # 计算总体指标
    overall_acc = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='binary')
    overall_recall = recall_score(y_true, y_pred, average='binary')
    overall_f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"Overall Test Accuracy: {overall_acc:.2%}")
    print(f"Overall Test Precision: {overall_precision:.2%}")
    print(f"Overall Test Recall: {overall_recall:.2%}")
    print(f"Overall Test F1-Score: {overall_f1:.2%}")

def result_test(real, pred):
    cv_conf = confusion_matrix(real, pred)
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='binary')
    recall = recall_score(real, pred, average='binary')
    f1 = f1_score(real, pred, average='binary')
    patten = 'test:  acc: {:.4f}   precision: {:.4f}   recall: {:.4f}   f1: {:.4f}'
    print(patten.format(acc, precision, recall, f1))
    labels11 = ['negative', 'active']
    disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
    disp.plot(cmap="Blues", values_format='')
    plt.savefig("results/reConfusionMatrix.tif", dpi=400)
    plt.close()

# 模型评估
def dev_eval(model, data, loss_function, Result_test=False):
    '''
    :param model: 模型
    :param data: 验证集集或者测试集的数据
    :param loss_function: 损失函数
    :param Result_test: 是否是测试集，用于生成混淆矩阵和ROC曲线
    :return: 损失、准确率、精确度、召回率、F1得分, 以及真实标签和预测标签（仅测试集）
    '''
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    y_pred_prob = []
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels)
            # 获取预测概率
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_pred_prob.extend(probs)
    
    acc = metrics.accuracy_score(labels_all, predict_all)
    precision = metrics.precision_score(labels_all, predict_all, average='binary')
    recall = metrics.recall_score(labels_all, predict_all, average='binary')
    f1 = metrics.f1_score(labels_all, predict_all, average='binary')
    
    if Result_test:
        result_test(labels_all, predict_all)
        return acc, loss_total / len(data), precision, recall, f1, labels_all, predict_all
    else:
        return acc, loss_total / len(data), precision, recall, f1

if __name__ == '__main__':
    # 设置随机数种子，保证每次运行结果一致，不至于不能复现模型
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = get_data()
    
    # 进行探索性数据分析（EDA）
    perform_eda(train_data, dev_data, test_data)  # Added EDA
    
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, shuffle=True),
        'dev': DataLoader(TextDataset(dev_data), batch_size, shuffle=True),
        'test': DataLoader(TextDataset(test_data), batch_size, shuffle=False)  # Changed shuffle to False for test
    }
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Model().to(device)
    init_network(model)
    
    # 添加交叉验证
    # 由于深度学习模型计算资源消耗大，这里采用在训练集内部进行分层交叉验证的方法
    # 这里只展示如何进行单次训练与评估，完整的K-fold需要重复训练
    # 可以参考下面的代码结构进行扩展
    
    # train(model, dataloaders)  # Original single training call

    # Implementing Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X = [x for x, y in train_data]
    y = [y for x, y in train_data]
    X = np.array(X)
    y = np.array(y)
    
    fold = 1
    results = []
    for train_index, val_index in skf.split(X, y):
        print(f"Starting Fold {fold}...")
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # 创建DataLoader
        train_fold = [(X_train_fold[i], y_train_fold[i]) for i in range(len(X_train_fold))]
        val_fold = [(X_val_fold[i], y_val_fold[i]) for i in range(len(X_val_fold))]
        
        train_loader = DataLoader(TextDataset(train_fold), batch_size, shuffle=True)
        val_loader = DataLoader(TextDataset(val_fold), batch_size, shuffle=True)
        
        current_dataloaders = {
            'train': train_loader,
            'dev': val_loader,
            'test': dataloaders['test']  # 使用原始测试集
        }
        
        # 重置模型参数
        model.apply(init_network)
        
        # 训练模型
        train(model, current_dataloaders)
        
        # 评估模型
        # 可以在此处收集每个fold的结果
        # For simplicity, assume the train function prints the results
        
        fold += 1
    
    # After cross-validation, perform final training on entire train set and evaluate on test set
    print("Training on the entire training set...")
    full_train_loader = DataLoader(TextDataset(train_data), batch_size, shuffle=True)
    full_val_loader = DataLoader(TextDataset(dev_data), batch_size, shuffle=True)
    
    full_dataloaders = {
        'train': full_train_loader,
        'dev': full_val_loader,
        'test': dataloaders['test']
    }
    
    # 重置模型参数
    model.apply(init_network)
    
    # 训练模型
    train(model, full_dataloaders)
    
    print("Training and evaluation completed.")
