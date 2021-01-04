import string
from nltk.corpus import stopwords
import nltk
from gensim.models.word2vec import LineSentence, Word2Vec
import time
from nltk.stem import PorterStemmer
from preprocess import fin_deal

#定义不需要的词性
tags_not = ['CD', 'CC', 'IN', 'MD', 'UH', 'WDT', 'WP', 'DT']
model_path = "model/{}/w2v_{}.mod"
des_path = "text/{}/api_des_{}.txt"


def desPrecessing(text):
    text = text.lower()  # 注意要text=。。。
    for c in string.punctuation:
        text = text.replace(c, ' ')
    wordlist = nltk.word_tokenize(text)  # 分词
    filtered = [w for w in wordlist if w not in stopwords.words('english')]  # 去除停用词
    refiltered = nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos not in tags_not]  # 仅保留名词或特定pos，去除数词
    ps = PorterStemmer()  # 词干化
    filtered = [ps.stem(w) for w in filtered]
    return " ".join(filtered)


def save_coprus(Api_dic, cnt_cata):
    start = time.perf_counter()
    doclist_1 = []
    for api in Api_dic:
        Api_describe = Api_dic[api]['Description']
        description_all = str(Api_describe)
        doclist_1.append(desPrecessing(description_all))

    with open(des_path.format(cnt_cata, cnt_cata), "w", encoding='utf-8') as f_ps:
        for line in doclist_1:
            f_ps.write(line + "\n")
    f_ps.close()

    elapsed = (time.perf_counter() - start)
    print("save_coprus() Time used:", elapsed)


def save_model(cnt_cata):

    start = time.perf_counter()
    sentences = LineSentence(des_path.format(cnt_cata, cnt_cata))
    mo = Word2Vec(sentences,  min_count=2, iter=1000)  # 迭代次数？？？
    mo.save(model_path.format(cnt_cata, cnt_cata))
    print("模型构建完成")

    model_w2v = Word2Vec.load(model_path.format(cnt_cata, cnt_cata))
    candidates = []
    with open(des_path.format(cnt_cata, cnt_cata), encoding="utf-8") as f:
        for line in f:
            candidates.append(line.strip().split())  # strip()去除首尾空格；split()通过指定分隔符对字符串进行切片，默认空格

    cnt_des = []
    res_data = []
    for candidate in candidates:
        single = []
        for c in candidate:
            if c in model_w2v.wv.vocab:
                single.append(c)
                if c not in cnt_des:
                    cnt_des.append(c)
        res_data.append(" ".join(single))
    with open(des_path.format(cnt_cata, cnt_cata), "w", encoding='utf-8') as f_ps:
        for line in res_data:
            f_ps.write(line + "\n")
    f_ps.close()

    print("特征数：", len(cnt_des))
    elapsed = (time.perf_counter() - start)
    print("save_model() Time used:", elapsed)


if __name__ == "__main__":
    Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata = fin_deal(5)
    cnt_cata = 5
    # save_coprus(Api_dic, cnt_cata)
    save_model(cnt_cata)
