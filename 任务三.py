import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from tqdm import tqdm
import time


# 设置随机种子确保结果可复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


set_seed()


# 自定义数据集类
class RedditDataset(Dataset):
    def __init__(self, texts, emotions, labels, tokenizer, max_len=512):
        self.texts = texts
        self.emotions = emotions
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        emotion = self.emotions[idx]
        label = self.labels[idx]

        # 对文本进行tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'emotion': torch.tensor(emotion, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 注意力机制模块
class AttentionModule(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, text_features, emotion_features):
        # 拼接文本和情感特征
        combined = torch.cat([text_features, emotion_features], dim=1)

        # 计算注意力权重
        attention_weights = self.attention(combined)

        # 加权融合特征
        weighted_text = text_features * attention_weights
        weighted_emotion = emotion_features * (1 - attention_weights)

        # 合并加权后的特征
        fused_features = weighted_text + weighted_emotion

        return fused_features


# 多模态融合模型
class MultimodalFusionModel(nn.Module):
    def __init__(self, bert_model, emotion_dim, num_classes, dropout_rate=0.1):
        super(MultimodalFusionModel, self).__init__()
        self.bert = bert_model
        self.bert_hidden_size = bert_model.config.hidden_size
        self.emotion_dim = emotion_dim

        # 文本特征处理层
        self.text_projection = nn.Linear(self.bert_hidden_size, self.bert_hidden_size // 2)

        # 情感特征处理层
        self.emotion_projection = nn.Linear(emotion_dim, self.bert_hidden_size // 2)

        # 注意力融合模块
        self.attention_fusion = AttentionModule(self.bert_hidden_size // 2)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size // 2, self.bert_hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert_hidden_size // 4, num_classes)
        )

    def forward(self, input_ids, attention_mask, emotion_features):
        # 获取BERT文本特征
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output  # [CLS] token的输出

        # 投影文本特征
        text_features = self.text_projection(text_features)

        # 投影情感特征
        emotion_features = self.emotion_projection(emotion_features)

        # 使用注意力机制融合特征
        fused_features = self.attention_fusion(text_features, emotion_features)

        # 分类
        logits = self.classifier(fused_features)

        return logits


# 数据预处理函数
def preprocess_data(json_data, emotion_mapping):
    texts = []
    emotions = []
    labels = []

    for key, item in json_data.items():
        # 提取文本
        text = item['Reddit Post']
        texts.append(text)

        # 提取情感特征
        emotion_vector = np.zeros(len(emotion_mapping))
        for annotation_id, annotations in item['Annotations'].items():
            for annotation in annotations:
                emotion = annotation['Emotion']
                if emotion in emotion_mapping:
                    emotion_vector[emotion_mapping[emotion]] = 1

        emotions.append(emotion_vector)

        # 这里假设我们要预测的标签是一个二分类问题
        # 例如：根据文本和情感特征预测是否为积极情感
        # 这需要根据实际任务调整
        # 简单示例：如果包含"joy"情感，标记为1，否则为0
        label = 1 if emotion_mapping.get('joy', 0) in np.where(emotion_vector == 1)[0] else 0
        labels.append(label)

    return texts, emotions, labels


# 训练函数
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)

    # 记录训练开始时间
    start_time = time.time()

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }

    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 50}")

        # 训练模式
        model.train()
        train_loss = 0.0

        # 使用tqdm显示训练进度条
        train_progress = tqdm(train_dataloader, desc="Training", unit="batch")

        for batch_idx, batch in enumerate(train_progress):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion = batch['emotion'].to(device)
            labels = batch['label'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(input_ids, attention_mask, emotion)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 更新损失
            train_loss += loss.item()

            # 更新进度条信息
            avg_loss = train_loss / (batch_idx + 1)
            train_progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)

        # 验证模式
        print("\nValidating...")
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        # 使用tqdm显示验证进度条
        val_progress = tqdm(val_dataloader, desc="Validation", unit="batch")

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_progress):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                emotion = batch['emotion'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, emotion)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # 获取预测结果
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                # 更新进度条信息
                avg_loss = val_loss / (batch_idx + 1)
                val_progress.set_postfix({"loss": f"{avg_loss:.4f}"})

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)

        # 计算评估指标
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')

        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)

        # 计算当前epoch耗时
        epoch_time = time.time() - start_time
        hours, rem = divmod(epoch_time, 3600)
        minutes, seconds = divmod(rem, 60)

        # 打印当前epoch的详细信息
        print(f"\nTraining Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation F1 Score: {val_f1:.4f}")
        print(f"Time elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        print(f"{'=' * 50}")

    # 计算总训练时间
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTraining completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    return model, history


# 评估函数
def evaluate_model(model, test_dataloader, criterion, device):
    model.to(device)
    model.eval()

    print("\nEvaluating on test set...")
    test_loss = 0.0
    all_preds = []
    all_labels = []

    # 使用tqdm显示测试进度条
    test_progress = tqdm(test_dataloader, desc="Testing", unit="batch")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_progress):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emotion = batch['emotion'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, emotion)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条信息
            avg_loss = test_loss / (batch_idx + 1)
            test_progress.set_postfix({"loss": f"{avg_loss:.4f}"})

    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_dataloader)

    # 计算评估指标
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print('Classification Report:')
    print(report)

    return avg_test_loss, test_accuracy, test_f1, report


# 保存训练历史记录
def save_history(history, output_dir):
    import json
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {os.path.join(output_dir, 'training_history.json')}")


# 主函数
def main():
    # 设置路径
    bert_path = r"C:\BERT"
    train_data_path = r"F:\Python\python_work\云计算与大数据分析\大作业\CovidET\train_anonymized-WITH_POSTS.json"
    val_data_path = r"F:\Python\python_work\云计算与大数据分析\大作业\CovidET\val_anonymized-WITH_POSTS.json"
    test_data_path = r"F:\Python\python_work\云计算与大数据分析\大作业\CovidET\test_anonymized-WITH_POSTS.json"

    # 定义情感映射
    emotion_mapping = {
        'disgust': 0,
        'sadness': 1,
        'joy': 2,
        'anger': 3
    }

    # 加载数据集
    print("Loading datasets...")
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)

    with open(val_data_path, 'r') as f:
        val_data = json.load(f)

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    # 预处理数据
    print("Preprocessing data...")
    train_texts, train_emotions, train_labels = preprocess_data(train_data, emotion_mapping)
    val_texts, val_emotions, val_labels = preprocess_data(val_data, emotion_mapping)
    test_texts, test_emotions, test_labels = preprocess_data(test_data, emotion_mapping)

    # 加载本地BERT tokenizer和模型
    print(f"Loading BERT model from {bert_path}...")
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path)

    # 创建数据集和数据加载器
    print("Creating data loaders...")
    train_dataset = RedditDataset(train_texts, train_emotions, train_labels, tokenizer)
    val_dataset = RedditDataset(val_texts, val_emotions, val_labels, tokenizer)
    test_dataset = RedditDataset(test_texts, test_emotions, test_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # 创建多模态融合模型
    print("Initializing model...")
    emotion_dim = len(emotion_mapping)
    num_classes = 2  # 二分类任务
    model = MultimodalFusionModel(bert_model, emotion_dim, num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 训练模型
    print("\nStarting training...")
    trained_model, history = train_model(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        device,
        epochs=5
    )

    # 评估模型
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy, test_f1, report = evaluate_model(trained_model, test_dataloader, criterion, device)

    # 保存模型
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "multimodal_fusion_model.pt")
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # 保存训练历史
    save_history(history, model_dir)


if __name__ == "__main__":
    main()
