import json
import requests
import pandas as pd
from tqdm import tqdm
import re
import time
import random
import os

# 数据文件路径
DATA_PATH = r"F:\Python\python_work\云计算与大数据分析\大作业\twitter_dataset\testset\posts_groundtruth.txt"

# 抽样数量
SAMPLE_SIZE = 50

# API配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # 可根据实际模型名称修改
MAX_RETRIES = 5  # 增加重试次数
INITIAL_WAIT_TIME = 2  # 初始等待时间(秒)
MAX_WAIT_TIME = 30  # 最大等待时间(秒)

# 结果保存路径
RESULT_DIR = r"F:\Python\python_work\云计算与大数据分析\大作业\results"
os.makedirs(RESULT_DIR, exist_ok=True)  # 确保目录存在


# 调用ollama模型（增强版，带指数退避重试）
def call_llama3(prompt, model=MODEL_NAME, max_retries=MAX_RETRIES):
    wait_time = INITIAL_WAIT_TIME
    retries = 0

    while retries < max_retries:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # 降低随机性，提高结果一致性
                    "max_tokens": 50  # 限制生成token数量，避免过长响应
                }
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=30)

            if response.status_code == 200:
                try:
                    result = response.json()
                    # 兼容不同版本ollama的响应格式
                    if "response" in result:
                        answer = result["response"].strip()
                    elif "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0].get("text", "").strip()
                    else:
                        # 无法解析有效回答，打印详细响应用于调试
                        print(f"响应格式异常: {result}")
                        retries += 1
                        wait_time = min(wait_time * 2, MAX_WAIT_TIME)  # 指数退避
                        time.sleep(wait_time)
                        continue

                    # 打印实际响应（用于调试）
                    print(f"模型响应: {answer[:50]}..." if len(answer) > 50 else f"模型响应: {answer}")

                    # 灵活匹配结果（增加对模糊回答的处理）
                    if re.search(r'0|假|fake|false|否|负面|消极', answer, re.IGNORECASE):
                        return 0
                    elif re.search(r'1|真|real|true|是|正面|积极', answer, re.IGNORECASE):
                        return 1
                    elif re.search(r'2|中|中性|客观', answer, re.IGNORECASE):
                        return 2  # 情感分析的"中性"
                    else:
                        # 无法识别结果，打印详细回答用于调试
                        print(f"无法识别的回答: {answer}")
                        retries += 1
                        wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                        time.sleep(wait_time)
                        continue
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"JSON解析错误: {str(e)}，响应内容: {response.text[:100]}...")
                    retries += 1
                    wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                    time.sleep(wait_time)
            else:
                print(f"HTTP错误，状态码: {response.status_code}，响应内容: {response.text[:100]}...")
                retries += 1
                wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                time.sleep(wait_time)
        except requests.exceptions.Timeout:
            print("请求超时，等待后重试...")
            retries += 1
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
            time.sleep(wait_time)
        except requests.exceptions.ConnectionError:
            print("连接错误，检查ollama服务是否运行...")
            retries += 1
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
            time.sleep(wait_time)
        except Exception as e:
            print(f"调用异常: {str(e)}")
            retries += 1
            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
            time.sleep(wait_time)

    print(f"多次调用失败（{max_retries}次），放弃处理当前prompt")
    return None


# 解析数据集
def parse_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        lines = [line.strip() for line in f if line.strip()]

        # 随机抽样
        if len(lines) > SAMPLE_SIZE:
            print(f"从{len(lines)}条数据中随机抽取{SAMPLE_SIZE}条...")
            lines = random.sample(lines, SAMPLE_SIZE)

        for line in tqdm(lines, desc="解析抽样数据"):
            parts = line.split('\t')
            if len(parts) >= 7:
                post_id = parts[0]
                post_text = '\t'.join(parts[1:6])
                label = parts[-1] if len(parts) >= 7 else "fake"
                data.append({
                    'post_id': post_id,
                    'post_text': post_text,
                    'label': 1 if label.lower() == 'real' else 0
                })
    return pd.DataFrame(data)


# 任务1：设计基础判别prompt（简化版，减少token消耗）
def task1_prompt(text):
    # 限制文本长度，避免过长prompt
    max_text_length = 500
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    return f"新闻真假判断：0=假新闻，1=真新闻。仅回复数字：\n{text}"


# 任务1：执行真假判别并统计准确率
def task1_accuracy(df):
    predictions = []
    for i, text in enumerate(tqdm(df['post_text'], desc="任务1：预测真假新闻")):
        print(f"\n处理第{i + 1}/{len(df)}条新闻 (ID: {df.iloc[i]['post_id']})...")
        prompt = task1_prompt(text)
        pred = call_llama3(prompt)
        predictions.append(pred if pred is not None else -1)

        # 添加请求间隔，避免服务过载
        time.sleep(1)

    df['task1_pred'] = predictions
    valid_df = df[df['task1_pred'] != -1]

    total = len(valid_df)
    if total == 0:
        return {"accuracy": 0, "accuracy_fake": 0, "accuracy_true": 0, "valid_samples": 0}

    correct = (valid_df['label'] == valid_df['task1_pred']).sum()
    fake_total = (valid_df['label'] == 0).sum()
    fake_correct = ((valid_df['label'] == 0) & (valid_df['task1_pred'] == 0)).sum()
    real_total = (valid_df['label'] == 1).sum()
    real_correct = ((valid_df['label'] == 1) & (valid_df['task1_pred'] == 1)).sum()

    return {
        "accuracy": correct / total,
        "accuracy_fake": fake_correct / fake_total if fake_total > 0 else 0,
        "accuracy_true": real_correct / real_total if real_total > 0 else 0,
        "valid_samples": total
    }


# 任务2：设计情感分析prompt（简化版）
def task2_prompt(text):
    max_text_length = 500
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    return f"情感分析：0=负面，1=中性，2=正面。仅回复数字：\n{text}"


# 任务2：执行情感分析
def task2_sentiment(df):
    sentiments = []
    for i, text in enumerate(tqdm(df['post_text'], desc="任务2：分析情感倾向")):
        print(f"\n处理第{i + 1}/{len(df)}条新闻 (ID: {df.iloc[i]['post_id']})...")
        prompt = task2_prompt(text)
        sent = call_llama3(prompt)
        sentiments.append(sent if sent is not None else -1)

        # 添加请求间隔
        time.sleep(1)

    valid_indices = [i for i, s in enumerate(sentiments) if s != -1]
    return [sentiments[i] for i in valid_indices], valid_indices


# 任务3：设计结合情感的判别prompt
def task3_prompt(text, sentiment):
    max_text_length = 500
    if len(text) > max_text_length:
        text = text[:max_text_length] + "..."

    sentiment_desc = ["负面", "中性", "正面"][sentiment]
    return f"情感倾向：{sentiment_desc}。新闻真假判断：0=假新闻，1=真新闻。仅回复数字：\n{text}"


# 任务3：结合情感执行真假判别并统计准确率
def task3_accuracy(df, valid_indices, sentiments):
    df_valid = df.iloc[valid_indices].copy()
    df_valid['sentiment'] = sentiments

    predictions = []
    for i, row in enumerate(tqdm(df_valid.iterrows(), desc="任务3：结合情感预测真假新闻")):
        print(f"\n处理第{i + 1}/{len(df_valid)}条新闻 (ID: {row[1]['post_id']})...")
        prompt = task3_prompt(row[1]['post_text'], row[1]['sentiment'])
        pred = call_llama3(prompt)
        predictions.append(pred if pred is not None else -1)

        # 添加请求间隔
        time.sleep(1)

    df_valid['task3_pred'] = predictions
    valid_df = df_valid[df_valid['task3_pred'] != -1]

    total = len(valid_df)
    if total == 0:
        return {"accuracy": 0, "accuracy_fake": 0, "accuracy_true": 0, "valid_samples": 0}

    correct = (valid_df['label'] == valid_df['task3_pred']).sum()
    fake_total = (valid_df['label'] == 0).sum()
    fake_correct = ((valid_df['label'] == 0) & (valid_df['task3_pred'] == 0)).sum()
    real_total = (valid_df['label'] == 1).sum()
    real_correct = ((valid_df['label'] == 1) & (valid_df['task3_pred'] == 1)).sum()

    return {
        "accuracy": correct / total,
        "accuracy_fake": fake_correct / fake_total if fake_total > 0 else 0,
        "accuracy_true": real_correct / real_total if real_total > 0 else 0,
        "valid_samples": total
    }


# 主函数：整合所有任务
def main():
    # 1. 解析数据并抽样
    print("开始解析数据并抽样...")
    df = parse_data(DATA_PATH)
    print(f"数据抽样完成，共{len(df)}条新闻，其中真新闻{df['label'].sum()}条，假新闻{len(df) - df['label'].sum()}条")

    if len(df) == 0:
        print("抽样后无有效样本，程序退出")
        return

    # 2. 任务1：基础真假判别
    print("\n执行任务1：仅用prompt判别新闻真假...")
    task1_predictions = []
    for i, text in enumerate(tqdm(df['post_text'], desc="任务1：预测真假新闻")):
        print(f"\n处理第{i + 1}/{len(df)}条新闻 (ID: {df.iloc[i]['post_id']})...")
        prompt = task1_prompt(text)
        pred = call_llama3(prompt)
        task1_predictions.append(pred if pred is not None else -1)
        time.sleep(1)  # 避免请求过快

    df['task1_pred'] = task1_predictions
    valid_df1 = df[df['task1_pred'] != -1]

    task1_results = {
        "accuracy": (valid_df1['label'] == valid_df1['task1_pred']).sum() / len(valid_df1) if len(valid_df1) > 0 else 0,
        "accuracy_fake": ((valid_df1['label'] == 0) & (valid_df1['task1_pred'] == 0)).sum() / (
                    valid_df1['label'] == 0).sum() if (valid_df1['label'] == 0).sum() > 0 else 0,
        "accuracy_true": ((valid_df1['label'] == 1) & (valid_df1['task1_pred'] == 1)).sum() / (
                    valid_df1['label'] == 1).sum() if (valid_df1['label'] == 1).sum() > 0 else 0,
        "valid_samples": len(valid_df1)
    }

    print(f"任务1准确率（有效样本数: {task1_results['valid_samples']}）：")
    print(f"总准确率: {task1_results['accuracy']:.4f}")
    print(f"假新闻准确率: {task1_results['accuracy_fake']:.4f}")
    print(f"真新闻准确率: {task1_results['accuracy_true']:.4f}")

    # 3. 任务2：情感分析
    print("\n执行任务2：分析新闻情感倾向...")
    task2_sentiments = []
    for i, text in enumerate(tqdm(df['post_text'], desc="任务2：分析情感倾向")):
        print(f"\n处理第{i + 1}/{len(df)}条新闻 (ID: {df.iloc[i]['post_id']})...")
        prompt = task2_prompt(text)
        sent = call_llama3(prompt)
        task2_sentiments.append(sent if sent is not None else -1)
        time.sleep(1)

    valid_indices = [i for i, s in enumerate(task2_sentiments) if s != -1]
    valid_sentiments = [task2_sentiments[i] for i in valid_indices]

    print(f"情感分析成功样本数: {len(valid_sentiments)}/{len(df)}")

    # 4. 任务3：结合情感判别真假
    if len(valid_indices) > 0:
        print("\n执行任务3：结合情感分析判别新闻真假...")
        df_valid = df.iloc[valid_indices].copy()
        df_valid['sentiment'] = valid_sentiments

        task3_predictions = []
        for i, row in enumerate(tqdm(df_valid.iterrows(), desc="任务3：结合情感预测真假新闻")):
            print(f"\n处理第{i + 1}/{len(df_valid)}条新闻 (ID: {row[1]['post_id']})...")
            prompt = task3_prompt(row[1]['post_text'], row[1]['sentiment'])
            pred = call_llama3(prompt)
            task3_predictions.append(pred if pred is not None else -1)
            time.sleep(1)

        df_valid['task3_pred'] = task3_predictions
        valid_df3 = df_valid[df_valid['task3_pred'] != -1]

        task3_results = {
            "accuracy": (valid_df3['label'] == valid_df3['task3_pred']).sum() / len(valid_df3) if len(
                valid_df3) > 0 else 0,
            "accuracy_fake": ((valid_df3['label'] == 0) & (valid_df3['task3_pred'] == 0)).sum() / (
                        valid_df3['label'] == 0).sum() if (valid_df3['label'] == 0).sum() > 0 else 0,
            "accuracy_true": ((valid_df3['label'] == 1) & (valid_df3['task3_pred'] == 1)).sum() / (
                        valid_df3['label'] == 1).sum() if (valid_df3['label'] == 1).sum() > 0 else 0,
            "valid_samples": len(valid_df3)
        }

        print(f"任务3准确率（有效样本数: {task3_results['valid_samples']}）：")
        print(f"总准确率: {task3_results['accuracy']:.4f}")
        print(f"假新闻准确率: {task3_results['accuracy_fake']:.4f}")
        print(f"真新闻准确率: {task3_results['accuracy_true']:.4f}")

        # 5. 对比结果
        print("\n准确率提升分析：")
        if task1_results['valid_samples'] > 0 and task3_results['valid_samples'] > 0:
            print(f"总准确率提升: {task3_results['accuracy'] - task1_results['accuracy']:.4f}")
            print(f"假新闻准确率提升: {task3_results['accuracy_fake'] - task1_results['accuracy_fake']:.4f}")
            print(f"真新闻准确率提升: {task3_results['accuracy_true'] - task1_results['accuracy_true']:.4f}")
        else:
            print("由于任务1或任务3无有效样本，无法进行对比")
    else:
        print("\n情感分析无有效样本，无法执行任务3")


if __name__ == "__main__":
    main()