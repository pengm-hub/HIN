import numpy as np
from preprocess import fin_deal
import time
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from gensim.models.word2vec import Word2Vec

model_path = "model/{}/w2v_{}.mod"
des_path = "text/{}/api_des_{}.txt"


def save_data(Api_dic, Api_id, Api_cata, Mashup_dic, Mashup_id, rel_apis, Tag_id, cnt_cata):
    data = {}

    data['ADES'] = w2v_sim(Api_dic, Api_id, cnt_cata)
    data['ATA'] = meta_ata(Api_dic, Api_id, Tag_id)
    data['AMAMA'] = meta_amama(Api_dic, Api_id, Mashup_dic, Mashup_id, rel_apis)

    adasum = 0
    matfilename = "ADA"
    data['ADA'] = np.zeros([len(Api_dic), len(Api_dic)])
    readata = sio.loadmat("matfile/{}/{}.mat".format(cnt_cata, matfilename))
    for i in range(readata[matfilename].shape[0]):
        for j in range(readata[matfilename].shape[1]):
            if readata[matfilename][i][j] >= 1:
                data['ADA'][i][j] = 1
                adasum = adasum + 1

    # features
    data['feature'] = api_fea(cnt_cata)

    # labels
    Cata = []
    for api in Api_dic:
        Cata.append(Api_cata.index(Api_dic[api]['Primary_Category']))
    Cata = np.array(Cata)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = Cata.reshape(len(Cata), 1)
    data['label'] = onehot_encoder.fit_transform(integer_encoded)

    # index
    data_index = list(range(5428))  # 注意加list
    print(data_index)
    X_test_vali, X_train, y_test_vali, y_train = train_test_split(data_index, data['label'], test_size=0.80, random_state=0)
    X_test, X_vali, _, _ = train_test_split(X_test_vali, y_test_vali, test_size=0.5, random_state=0)
    data['train_idx'] = np.array(X_train)
    data['val_idx'] = np.array(X_vali)
    data['test_idx'] = np.array(X_test)

    # save
    sio.savemat("matfile/{}/Apis_811.mat".format(cnt_cata), data)


def meta_amama(Api_dic, Api_id, Mashup_dic, Mashup_id, rel_apis):
    start = time.perf_counter()

    amamasum = 0
    am = np.zeros([len(Api_dic), len(Mashup_id)])
    ma = np.zeros([len(Mashup_id), len(Api_dic)])
    for m in Mashup_dic:
        rel = rel_apis[m]
        for api in rel:
            if api in Api_id:
                inda = Api_id.index(api)
                indm = Mashup_id.index(m)
                am[inda][indm] += 1
                ma[indm][inda] += 1
    mam = np.matmul(ma, am)
    amama = np.matmul(np.matmul(am, mam), ma)
    M_amama = np.zeros([len(Api_dic), len(Api_dic)])
    save_amama = np.zeros([len(Api_dic), len(Api_dic)])
    for i in range(amama.shape[0]):
        for j in range(amama.shape[1]):
            if (amama[i][i]+amama[j][j]) == 0.0:
                sim = 0.0
            else:
                sim = (2.000*amama[i][j])/(amama[i][i]+amama[j][j])
            # if sim > 0.0:
            #     print(sim)
            save_amama[i][j] = round(sim, 3)
            if sim > 0:
                M_amama[i][j] = 1
                amamasum += 1
    # ind = [(r1, r2) for r1 in rel_apis[m1] for r2 in rel_apis[m2] if r1 in Api_id and r2 in Api_id]
    print("amama边数:", amamasum)
    # save_mat(save_amama, "AMAMA")

    print("meta_amama():", time.perf_counter()-start)
    print("=========================================")
    return M_amama


def meta_ata(Api_dic, Api_id, Tag_id):
    start = time.perf_counter()
    atasum = 0
    at = np.zeros([len(Api_dic), len(Tag_id)])
    ta = np.zeros([len(Tag_id), len(Api_dic)])
    for api in Api_dic:
        tag = Api_dic[api]['Secondary_Categories'].split(',')
        for t in tag:
            if t in Tag_id:
                at[Api_id.index(api)][Tag_id.index(t)] = 1
                ta[Tag_id.index(t)][Api_id.index(api)] = 1
    ata = np.matmul(at, ta)
    M_ata = np.zeros([len(Api_dic), len(Api_dic)])
    # save_ata = np.zeros([len(Api_dic), len(Api_dic)])
    for i in range(ata.shape[0]):
        for j in range(ata.shape[1]):
            sim = (2.000*ata[i][j])/(ata[i][i]+ata[j][j])
            # save_ata = round(sim, 3)
            if sim >= 1:
                atasum += 1
                M_ata[i][j] = 1

    # save_mat(save_ata, "ATA")
    print("ata边数:", atasum)
    print("meta_ata():", time.perf_counter()-start)
    print("=========================================")
    return M_ata


def api_fea(cnt_cata):
    start = time.perf_counter()
    sens = []
    words = []
    with open(des_path.format(cnt_cata, cnt_cata), encoding="utf-8") as f:
        for line in f:
            word = line.strip().split()
            sens.append(word)
            for w in word:
                if w not in words:
                    words.append(w)
    csr = np.zeros((len(sens), len(words)))  # 你又忘记加两个括号了！！！
    cnt = 0
    for ss in sens:
        for s in ss:
            ind = words.index(s)
            csr[cnt][ind] = 1
        cnt += 1
    # save_mat(csr, "features")

    elapsed = (time.perf_counter() - start)
    print("sen_vec() Time used:", elapsed)
    return csr


def w2v_sim(Api_dic, Api_id, cnt_cata):
    start = time.perf_counter()
    model_w2v = Word2Vec.load(model_path.format(cnt_cata, cnt_cata))
    candidates = []
    M_des = np.zeros([len(Api_dic), len(Api_dic)])
    with open(des_path.format(cnt_cata, cnt_cata), encoding="utf-8") as f:
        for line in f:
            candidates.append(line.strip().split())  # strip()去除首尾空格；split()通过指定分隔符对字符串进行切片，默认空格

    for api in Api_dic:
        id = Api_id.index(api)
        words = candidates[id]
        word = []
        for w in words:
            word.append(w)
        res = []
        index = 0
        for candidate in candidates:
            if index <= id:
                index += 1
                continue
            score = model_w2v.n_similarity(word, candidate)
            results = {'id': index, "score": score}
            res.append(results)
            index += 1
        for i in range(len(res)):
            # fin_w2v[id][res[i]['id']] = res[i]['score']
            # fin_w2v[res[i]['id']][id] = res[i]['score']
            if res[i]['score'] > 0.8:
                M_des[id][res[i]['id']] = 1
                M_des[res[i]['id']][id] = 1

    elapsed = (time.perf_counter() - start)
    print("w2v_sim() Time used:", elapsed)
    return M_des


if __name__ == "__main__":
    Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata = fin_deal(5)
    save_data(Api_dic, Api_id, Api_cata, Mashup_dic, Mashup_id, rel_apis, Tag_id, 5)
    # api_fea()
    # meta_ata()
    # meta_amama()