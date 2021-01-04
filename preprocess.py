import csv
import copy
import numpy as np
import time


cata = [5, 10, 15, 20, 25, 30]
single_apis = [526, 368, 291, 254, 200, 165]


def Api_CSV(cnt_cata):
    start = time.perf_counter()
    Api_dic = {}
    Api_cata = []
    Api_id = []
    Tag_id = []
    # 获取到所有的信息
    with open("data/Api_Info.csv", 'r', encoding="UTF-8-sig") as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        name = ""
        for row in csv_reader:
            d = {}
            flag = 0
            for k, v in row.items():
                if k != "Secondary_Categories" and v == "":
                    flag = 1
                if "name" in k:
                    name = v
                    continue
                if k == "Primary_Category" and v not in Api_cata:
                    Api_cata.append(v)
                if k == "Secondary_Categories":
                    tags = v.split(',')
                    for t in tags:
                        if t not in Tag_id:
                            Tag_id.append(t)
                d[k] = v
            if flag == 0:
                Api_dic[name] = d
                Api_id.append(name)
    f.close()

    # 种类数
    catas = np.zeros(len(Api_cata))
    for api in Api_dic:
        c = Api_dic[api]["Primary_Category"]
        index = Api_cata.index(c)
        catas[index] += 1

    # 只要对应类别数的
    cnt = 0
    temp_cata = copy.copy(Api_cata)  # 一定要注意不能直接=，会指向同一个地址空间的，需要用copy
    for c in temp_cata:
        if catas[cnt] < single_apis[cata.index(cnt_cata)]: #  只要前N个类别的
            Api_cata.remove(c)
            catas = np.delete(catas, cnt)
            cnt -= 1
        cnt += 1

    for a in Api_id:
        c = Api_dic[a]["Primary_Category"]
        if c not in Api_cata:
            Api_dic.pop(a)

    Api_id.clear()
    for api in Api_dic:
        Api_id.append(api)

    print("剩余种类数：", len(Api_cata))
    print("剩余API数：", len(Api_dic))
    print("len(tag)", len(Tag_id))
    elapsed = (time.perf_counter() - start)
    print("Api_CSV() Time used:", elapsed)
    return Api_dic, Api_cata, Api_id, Tag_id


def Mashup_CSV():
    start = time.perf_counter()
    Mashup_dic = {}
    Mashup_id = []
    Mashup_Tag = []
    Mashup_cata = []
    with open("data/Mashup_Info.csv", 'r', encoding="UTF-8-sig") as f:
        reader = csv.reader(f)
        fieldnames = next(reader)
        csv_reader = csv.DictReader(f, fieldnames=fieldnames)
        name = ""
        for row in csv_reader:
            d = {}
            flag = 0
            for k, v in row.items():
                if v == "":
                    flag = 1
                if "name" in k:
                    name = v
                    continue
                if k == "Categories":
                    tags = v.split(',')
                    if tags[0] not in Mashup_cata:
                        Mashup_cata.append(tags[0])
                    for t in tags:
                        if t not in Mashup_Tag:
                            Mashup_Tag.append(t)
                d[k] = v
            if flag == 0:
                Mashup_dic[name] = d
                if name not in Mashup_id:  # 有同名mashup
                    Mashup_id.append(name)
    f.close()

    print("剩余种类数：", len(Mashup_cata))
    print("剩余Mashup数：", len(Mashup_dic))
    print("len(Mashup_tag)", len(Mashup_Tag))
    elapsed = (time.perf_counter() - start)
    print("Mashup_CSV() Time used:", elapsed)
    return Mashup_dic, Mashup_id, Mashup_Tag, Mashup_cata


def fin_deal(cnt_cata):
    start = time.perf_counter()
    Api_dic, Api_cata, Api_id, Tag_id = Api_CSV(cnt_cata)
    Mashup_dic, Mashup_id, Mashup_tag, Mashup_cata = Mashup_CSV()
    rel_apis = {}
    for ma in Mashup_dic:
        rel_api = Mashup_dic[ma]["Related_APIs"]
        rel_apis[ma] = rel_api.split(',')
        for api in rel_apis[ma]:
            if api not in Api_id:
                rel_apis[ma].remove(api)

    elapsed = (time.perf_counter() - start)
    print("fin_deal() Time used:", elapsed)
    return Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata


if __name__ == "__main__":
    # Api_CSV()
    # Mashup_CSV()
    Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata = fin_deal(20)
    # save_coprus()