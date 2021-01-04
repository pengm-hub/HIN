from preprocess import fin_deal
from w2v import save_coprus, save_model
from lda import cal_lda
from ApiConstruct import save_data

cata = [5, 10, 15, 20, 25, 30]
w2v_path = "model/w2v_{}.mod"
api_des = "text/{}/Api_Description.txt"


for i in cata:
    Api_dic, Mashup_dic, Mashup_id, rel_apis, Api_cata, Api_id, Tag_id, Mashup_tag, Mashup_cata = fin_deal(i)
    # save_coprus(Api_dic, i)
    # save_model(i)
    cal_lda(Api_dic, i)
    save_data(Api_dic, Api_id, Api_cata, Mashup_dic, Mashup_id, rel_apis, Tag_id, 5)

