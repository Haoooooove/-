import re
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models
import json
import requests
import time
import os
from tqdm import tqdm

# 确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 配置参数
DATA_PATH = r"F:\Python\python_work\云计算与大数据分析\大作业\twitter_dataset\testset\posts_groundtruth.txt"
RESULT_DIR = r"F:\Python\python_work\云计算与大数据分析\大作业\results\topic_analysis"
os.makedirs(RESULT_DIR, exist_ok=True)

# Ollama API配置
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"
MAX_RETRIES = 7
INITIAL_WAIT_TIME = 2
MAX_WAIT_TIME = 60


# 数据预处理函数
def preprocess_text(text):
    # 1. 去除非字母和中文的字符，保留英文和中文
    text = re.sub(r'[^a-zA-Z\u4e00-\u9fa5\s]', '', text.lower())

    # 2. 分词
    words = text.split()

    # 3. 去停用词和短词
    stop_words = set(stopwords.words('english'))
    chinese_stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
                         '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    stop_words.update(chinese_stopwords)
    words = [word for word in words if word not in stop_words and len(word) > 2]

    # 4. 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words


# 调用大模型分析主题
def analyze_topic_with_llm(topic_id, top_words, top_documents, analysis_path):
    # 断点续传检查
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            existing_analyses = json.load(f)
        if str(topic_id) in existing_analyses:
            return existing_analyses[str(topic_id)]

    # 构建prompt
    prompt = f"""
    分析以下Twitter主题：
    1. 核心内容
    2. 反映的社会现象
    3. 情感倾向
    4. 主题名称（不超过5个词）

    主题 #{topic_id} 关键词: {', '.join(top_words)}
    代表性推文:
    {top_documents}

    输出JSON格式：{{"core_content":"","social_phenomenon":"","sentiment":"","topic_name":""}}
    """

    retries = 0
    wait_time = INITIAL_WAIT_TIME

    while retries < MAX_RETRIES:
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 500, "timeout": 60}
            }
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)

            if response.status_code == 200:
                answer = response.json().get("response", "")
                json_match = re.search(r'\{.*\}', answer, re.DOTALL)
                if json_match:
                        try:
                            analysis = json.loads(json_match.group(0))
                            # 保存结果
                            if os.path.exists(analysis_path):
                                with open(analysis_path, 'r', encoding='utf-8') as f:
                                    existing_analyses = json.load(f)
                            else:
                                existing_analyses = {}

                            existing_analyses[str(topic_id)] = analysis
                            with open(analysis_path, 'w', encoding='utf-8') as f:
                                json.dump(existing_analyses, f, indent=4, ensure_ascii=False)

                            return analysis
                        except json.JSONDecodeError:
                            print(f"JSON解析失败: {answer[:100]}...")
                            retries += 1
                            wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                            time.sleep(wait_time)
                            continue
                else:
                    print(f"未找到JSON响应: {answer[:100]}...")
                    retries += 1
                    wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                    time.sleep(wait_time)
                    continue
            else:
                print(f"HTTP错误: {response.status_code}, 响应内容: {response.text[:100]}...")
                retries += 1
                wait_time = min(wait_time * 2, MAX_WAIT_TIME)
                time.sleep(wait_time)
        except requests.exceptions.Timeout:
            print(f"请求超时，等待{wait_time}秒后重试...")
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

    print(f"主题 {topic_id} 分析失败，返回默认结果")
    return {
        "core_content": "分析失败",
        "social_phenomenon": "分析失败",
        "sentiment": "分析失败",
        "topic_name": f"主题{topic_id}"
    }


# 主函数
def main():
    # 1. 数据准备（增加条件抽取有效样本）
    print("开始数据准备...")
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据文件 {DATA_PATH} 不存在")
        return

    try:
        # 读取数据，指定列名
        df = pd.read_csv(DATA_PATH, sep='\t', header=None,
                         names=['id', 'text1', 'text2', 'text3', 'text4', 'text5', 'label'])
        print(f"原始数据: {len(df)} 条记录")

        # 合并文本列，创建完整文本字段
        text_columns = ['text1', 'text2', 'text3', 'text4', 'text5']
        df['full_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

        # 过滤条件：
        # 1. 完整文本长度 > 10个字符（确保有足够内容）
        # 2. 至少有2个非空文本列（确保数据质量）
        valid_df = df[
            (df['full_text'].str.len() > 10) &
            (df[text_columns].count(axis=1) >= 2)
            ]

        print(f"有效数据: {len(valid_df)} 条记录")

        # 如果有效数据不足10条，使用所有有效数据
        SAMPLE_SIZE = min(10, len(valid_df))

        # 随机抽样10条有效数据
        sampled_df = valid_df.sample(SAMPLE_SIZE, random_state=42)
        print(f"抽样完成，共 {SAMPLE_SIZE} 条有效新闻")

        # 保存抽样数据，方便调试
        sampled_path = os.path.join(RESULT_DIR, 'sampled_data.csv')
        sampled_df.to_csv(sampled_path, index=False, encoding='utf-8')
        print(f"抽样数据已保存至: {sampled_path}")

    except Exception as e:
        print(f"数据准备错误: {str(e)}")
        return

    # 2. 数据预处理
    print("\n开始数据预处理...")
    try:
        # 应用预处理函数，添加处理后的文本列
        sampled_df['processed_text'] = sampled_df['full_text'].apply(preprocess_text)

        # 计算每条新闻处理后的词数，用于筛选
        sampled_df['word_count'] = sampled_df['processed_text'].apply(len)

        # 过滤掉处理后词数 < 5的新闻（确保有足够的词用于主题建模）
        filtered_df = sampled_df[sampled_df['word_count'] >= 5]

        # 如果过滤后数据不足10条，使用所有过滤后数据
        if len(filtered_df) < SAMPLE_SIZE:
            print(f"警告: 经过预处理后，只有 {len(filtered_df)} 条新闻包含足够的词（≥5个）")
            SAMPLE_SIZE = len(filtered_df)
            filtered_df = filtered_df.sample(SAMPLE_SIZE, random_state=42)

        print(f"预处理完成，保留 {SAMPLE_SIZE} 条有效新闻")

        # 保存预处理后的数据
        processed_path = os.path.join(RESULT_DIR, 'processed_data.csv')
        filtered_df[['id', 'full_text', 'processed_text', 'word_count']].to_csv(
            processed_path, index=False, encoding='utf-8'
        )
        print(f"预处理数据已保存至: {processed_path}")

    except Exception as e:
        print(f"数据预处理错误: {str(e)}")
        return

    # 3. 构建词典和语料库
    print("\n构建词典和语料库...")
    try:
        # 从处理后的文本创建词典
        dictionary = corpora.Dictionary(filtered_df['processed_text'])

        # 过滤极端频率的词（出现次数少于1或超过95%的文档）
        dictionary.filter_extremes(no_below=1, no_above=0.95)

        # 将文档转换为词袋表示
        corpus = [dictionary.doc2bow(text) for text in filtered_df['processed_text']]

        print(f"词典大小: {len(dictionary)} 个词")
        print(f"语料库大小: {len(corpus)} 个文档")

    except Exception as e:
        print(f"词典/语料库构建错误: {str(e)}")
        return

    # 4. 训练LDA模型
    print("\n训练LDA模型...")
    try:
        # 根据样本量动态调整主题数量（最多3个主题）
        num_topics = min(3, max(2, SAMPLE_SIZE // 5))
        print(f"设置主题数量: {num_topics}")

        # 训练LDA模型
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            alpha='auto',
            eta='auto'
        )

        # 保存LDA模型
        model_path = os.path.join(RESULT_DIR, 'lda_model')
        lda_model.save(model_path)
        print(f"LDA模型已保存至: {model_path}")

        # 打印每个主题的前10个关键词
        print("\n主题关键词:")
        for topic_id in range(num_topics):
            top_words = lda_model.show_topic(topic_id, topn=10)
            print(f"主题 #{topic_id + 1}: {', '.join([word for word, _ in top_words])}")

    except Exception as e:
        print(f"LDA模型训练错误: {str(e)}")
        return

    # 5. 可视化分析
    print("\n生成可视化分析...")
    try:
        # 5.1 pyLDAvis交互图
        vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        vis_path = os.path.join(RESULT_DIR, 'lda_visualization.html')
        pyLDAvis.save_html(vis_data, vis_path)
        print(f"pyLDAvis交互图已保存至: {vis_path}")

        # 5.2 词云图
        plt.figure(figsize=(15, 10))
        for i in range(num_topics):
            plt.subplot(int(num_topics / 2) + 1, 2, i + 1)

            # 获取主题词及其权重
            topic_terms = lda_model.get_topic_terms(i, topn=20)
            topic_dict = {dictionary[id]: weight for id, weight in topic_terms}

            # 生成词云
            wordcloud = WordCloud(
                background_color='white',
                max_words=100,
                width=800,
                height=400,
                font_path="C:/Windows/Fonts/simhei.ttf"  # 指定中文字体路径
            ).generate_from_frequencies(topic_dict)

            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'主题 #{i + 1}')
            plt.axis('off')

        wordcloud_path = os.path.join(RESULT_DIR, 'topic_wordclouds.png')
        plt.tight_layout()
        plt.savefig(wordcloud_path, dpi=300)
        print(f"主题词云图已保存至: {wordcloud_path}")

        # 5.3 文档-主题分布矩阵（热力图）
        doc_topic_matrix = np.zeros((len(filtered_df), num_topics))
        for i, doc in enumerate(corpus):
            topic_dist = lda_model.get_document_topics(doc)
            for topic_id, prob in topic_dist:
                doc_topic_matrix[i, topic_id] = prob

        plt.figure(figsize=(12, 8))
        plt.imshow(doc_topic_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='主题概率')
        plt.xlabel('主题')
        plt.ylabel('文档')
        plt.title('文档-主题分布热力图')

        # 添加x轴标签（主题ID）
        plt.xticks(range(num_topics), [f'主题{i + 1}' for i in range(num_topics)])

        # 添加y轴标签（文档ID的前8个字符）
        plt.yticks(range(len(filtered_df)), [str(filtered_df.iloc[i]['id'])[:8] for i in range(len(filtered_df))])

        heatmap_path = os.path.join(RESULT_DIR, 'doc_topic_heatmap.png')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        print(f"文档-主题热力图已保存至: {heatmap_path}")
    except Exception as e:
        print(f"可视化错误: {str(e)}")
        return

    # 6. 结合大模型分析主题
    print("\n使用大模型分析主题内容...")
    analysis_path = os.path.join(RESULT_DIR, 'topic_analyses.json')
    topic_analyses = {}

    for topic_id in range(1, num_topics + 1):
        print(f"\n分析主题 #{topic_id}...")

        # 获取主题的前15个关键词
        top_words = [dictionary[id] for id, _ in lda_model.get_topic_terms(topic_id - 1, topn=15)]

        # 获取属于该主题概率最高的3个文档
        topic_docs = []
        doc_probs = []

        for i, doc in enumerate(corpus):
            topic_dist = lda_model.get_document_topics(doc)
            topic_probs = {t_id: prob for t_id, prob in topic_dist}
            if (topic_id - 1) in topic_probs:
                doc_probs.append((i, topic_probs[topic_id - 1]))

        # 按概率排序并选取前3个
        doc_probs.sort(key=lambda x: x[1], reverse=True)
        selected_docs = []

        for i, _ in doc_probs[:3]:
            # 截取原文的前120个字符
            doc_text = filtered_df.iloc[i]['full_text'][:120] + "..." if len(
                filtered_df.iloc[i]['full_text']) > 120 else filtered_df.iloc[i]['full_text']
            selected_docs.append(f"文档 {filtered_df.iloc[i]['id']}: {doc_text}")

        # 拼接文档示例
        top_documents = "\n".join(selected_docs) if selected_docs else "无代表性文档"

        # 调用大模型分析
        analysis = analyze_topic_with_llm(topic_id, top_words, top_documents, analysis_path)
        topic_analyses[topic_id] = analysis

        # 打印分析结果
        print(f"主题 #{topic_id} 分析结果:")
        print(f"  主题名称: {analysis['topic_name']}")
        print(f"  核心内容: {analysis['core_content']}")
        print(f"  社会现象: {analysis['social_phenomenon']}")
        print(f"  情感倾向: {analysis['sentiment']}")

        # 添加请求间隔，避免API过载
        time.sleep(3)

    # 7. 保存主题分析结果
    try:
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(topic_analyses, f, indent=4, ensure_ascii=False)
        print(f"\n主题分析结果已保存至: {analysis_path}")
    except Exception as e:
        print(f"结果保存错误: {str(e)}")
        return

    print(f"所有分析结果已保存至目录: {RESULT_DIR}")
    print("主题分析任务完成！")


if __name__ == "__main__":
    main()