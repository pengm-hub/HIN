from gensim import models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import scipy.io as sio
import numpy as np
from preprocess import fin_deal

des_path = "text/{}/api_des_{}.txt"


# 因为lda模型参数里有个minimum_probability，默认值0.01。只好手动修改成和topic一致的维度
def len_process(doc_lda, cnt_cata):
    doc_lda_1 = []
    j = 0
    for i in range(cnt_cata):
        if j < len(doc_lda) and doc_lda[j][0] == i:
            doc_lda_1.append(doc_lda[j])
            j = j+1
            continue
        doc_lda_1.append((i, 0))
    return doc_lda_1


def get_top(des, dictionary, lda):
    doc_bow = dictionary.doc2bow(des)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    list_doc = [i[1] for i in len_process(doc_lda)] # 输出新文档的主题分布
    return list_doc


def save_ldamodel(dictionary, text_data, cnt_cata):

    corpus = [dictionary.doc2bow(text) for text in text_data]
    ldamodel = LdaModel(corpus, num_topics=cnt_cata, id2word=dictionary)
    # 查看主题
    for topic in ldamodel.print_topics():
        print(topic[1])
    ldamodel.save('model/{}/ADA.gensim'.format(cnt_cata), "wb")


def cal_lda(Api_dic, cnt_cata):
    text_data = []
    with open(des_path.format(cnt_cata, cnt_cata), encoding="utf-8") as f:
        for line in f:
            text_data.append(line.strip().split())  # strip()去除首尾空格；split()通过指定分隔符对字符串进行切片，默认空格
    dictionary = Dictionary(text_data)
    save_ldamodel(dictionary, text_data)

    nonzero = 0.0
    ada = np.zeros([len(Api_dic), len(Api_dic)])
    lda = models.ldamodel.LdaModel.load('model/{}/ADA.gensim'.format(cnt_cata))

    for api1 in range(len(Api_dic)-1):
        ada[api1][api1] = 1.0
        des1 = text_data[api1]
        list_doc1 = get_top(des1, dictionary, lda)

        for api2 in range(api1+1, len(Api_dic)):
            des2 = text_data[api2]
            list_doc2 = get_top(des2, dictionary, lda)

            sim = round(np.dot(list_doc1, list_doc2) / (np.linalg.norm(list_doc1) * np.linalg.norm(list_doc2)), 3)
            ada[api1][api2] = ada[api2][api1] = sim
            nonzero = nonzero + 1.0

    print("lda矩阵稀疏：", nonzero/(len(Api_dic)*len(Api_dic)))

    dic = {}
    dic["ADA"] = ada
    sio.savemat("matfile/{}/ADA.mat".format(cnt_cata), dic)

    print("保存成功！")


# test
if __name__ == '__main__':
    Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata = fin_deal(5)
    cal_lda(Api_dic, 5)