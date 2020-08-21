from behavior_view.NTN.ntn import NeuralTensorNetwork, Load_Data
from behavior_view.NTN.ntn_train import detach, shuffle, train
from behavior_view.news_crawler import get_naver_news, news_combine, update_news_preprocess
from behavior_view.text_preprocess import Text_preprocess
# from behavior_view.fine_tuning.bert_fine_tuning import BertFineTuning
import pandas as pd
import numpy as np
import torch
from torch import optim
from ast import literal_eval
import argparse, calendar
import time, json, sys, os

sys.path.insert(0, os.getcwd()+r'\behavior_view\NTN')

if __name__ == "__main__":

    txt = Text_preprocess()
    local_path = r'C:\Users\JSH\Desktop\Github\stock_prediction\\'
    db_path = r'G:\공유 드라이브\Boad ADV Stock\\'
    parser = argparse.ArgumentParser(description='Update')
    parser.add_argument('industries_txt', help= 'companies_txt file')
    parser.add_argument('companies_txt', help= 'companies_txt file')
    parser.add_argument('keywords_txt', help= 'keywords_txt file')
    parser.add_argument('stopwordstxt', help= 'stopwords txt file') ## 기업별로 다른가? 그냥 합쳐서 써도 상관없을꺼 같은데
    parser.add_argument('replacedicttxt', help= 'replace dict txt file')
    args = parser.parse_args()
  
    industries = open(args.industries_txt, encoding= 'utf-8').read().splitlines()
    companies = literal_eval(open(args.companies_txt, encoding='utf-8').read())
    keywords = literal_eval(open(args.keywords_txt, encoding='utf-8').read())
    stopwords = open(args.stopwordstxt, encoding= 'utf-8').read().splitlines()
    keywords = literal_eval(open(args.replacedicttxt, encoding='utf-8').read())

    ## Today date
    nyear = time.gmtime(time.time()).tm_year
    nmon = time.gmtime(time.time()).tm_mon
    nday = time.gmtime(time.time()).tm_mday

    ## Start date
    pdate = pd.read_excel(db_path + r"화장품\event_embbed_with_tech_no_코스맥스.xlsx")['date'].iloc[-1]
    pyear = pdate.year
    pmon = pdate.month
    pday = pdate.day

    year_range = [year for year in range(pyear, nyear+1)]

## Update Crawling File
    end_month = 13
    start_time = time.time()
    for industry in industries:
        for query in keywords[industry]:
            file_path = local_path + r'new\{}\{}\\'.format(industry, query)
            for year in year_range:
                if year == pyear:
                    mm = pmon  ## start month
                else: mm = 1

                for i in list(range(mm, end_month)):
                    month_info = calendar.monthrange(year, i) ## last date of month
                    days = month_info[1]

                    if i < 10:
                        mm = "0"+str(i)
                    else:
                        mm = str(i)

                    for day in list(range(1, days+1)):
                        if day < 10:
                            dd = "0"+str(day)
                        else:
                            dd = str(day)
                        get_naver_news(file_path, query, str(year), mm, dd)

## Text_preprocess for all combined files
            new_combined = update_news_preprocess(industry, query)
            new_combined = txt.stop_words(news_combine, stopwords, 3)
            new_combined.header = new_combined.header.apply(txt.clean_text)
            new_combined.header = new_combined.header.apply(txt.papago)
            new_combined.to_excel(db_path + r"\fine_tuning\bert\{}_translate.xlsx".format(query))
            past_combined = pd.read_excel(db_path + r"{}\crwaled_data\{}_translate.xlsx".format(industry, query))
            combined = pd.concat([past_combined, new_combined])
            combined.to_excel(db_path + r"{}\crwaled_data\{}_translate.xlsx".format(industry, query))

### SVO EXTRACT with StanfordOpenIE and concat with past_SVO
            new_svo = txt.SVO_extractor(new_combined)
            with open(r"G:\공유 드라이브\Boad ADV Stock\{}\svo_pos_{}.json".format(industry, query), mode='r', encoding='utf-8') as f:
                svo_pos = json.load(f)
            for key in svo_pos.keys():
                svo_pos[key] += new_svo[key] 
            with open(r"G:\공유 드라이브\Boad ADV Stock\{}\svo_pos_{}.json".format(industry, query), mode='w', encoding='utf-8') as f:
                json.dump(svo_pos, f)

### Extract Embedded Vector from Bert model  new에다가 새롭게 저장해놓고 써야겠다 svo_pos는 또 따로 합쳐두고
            new_svo = txt.svo_embedding(new_svo)
            torch.save(new_svo, db_path + r"fine_tuning\ntn\svo_pos_{}.pt".format(query))
            svo_pos = torch.load(db_path + r"{}\svo_pos_{}.pt".format(industry, query))
            for key in svo_pos.keys():
                svo_pos[key] += new_svo[key]
            torch.save(svo_pos, db_path + r"{}\svo_pos_{}.pt".format(industry, query))
                      
            print(f"{query} nlp data update complete")

      
### Fine tuning for our model per company
        batch_size = 1024
        regularization = 0.00001
        lr = 0.01
        device = "cuda"

        for industry in industries:
            for company in companies[industry]:
                model = torch.load(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_{}.pt".format(industry, company))
                keywordstxt = open(db_path + r"\{}\keyword_per_company\keyword_{}.txt".format(industry, company), encoding='utf-8').read().splitlines()
                svo_pos_vector = pd.DataFrame()
                for f in os.listdir(db_path + r"fine_tuning\ntn\\"):
                    if f[8:-3] in keywordstxt:
                        svo_pos_keyword = torch.load(db_path + r"fine_tuning\ntn\\" + f)                        
                        svo_pos_keyword = pd.DataFrame(svo_pos_keyword)
                        for col in ['s_pos', 'v_pos', 'o_pos']:
                            svo_pos_keyword[col] = svo_pos_keyword[col].apply(detach)
                        svo_pos_vector = pd.concat([svo_pos_vector, svo_pos_keyword])
                        del svo_pos_keyword
                 
                svo_pos_vector = svo_pos_vector.to_dict('list')

                shuffle(svo_pos_vector)

                model.to(device)
                data_loader = Load_Data(svo_pos_vector, batch_size)
                optimizer = optim.Adam(model.parameters(), lr=lr)

                epochs= 500
                early_stop = [np.inf, np.inf, np.inf]
                k = 0
                for epoch in range(epochs):
                    loss, ntn = train(epoch, lr, data_loader)
                    if early_stop[-1] > loss:
                        print(early_stop)
                        early_stop.pop(0)
                        early_stop.append(loss)
                        k = 0
                        print(early_stop)
                        torch.save(ntn, r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_{}_{}.pt".format(industry, company, nyear+'_'+nmon+'_'+nday))
                    k += 1
                    if (k>3) & (epoch > 40):
                        break    

                print(f"{company} tensor network update complete")

        end_time = time.time()  
        print("*** 전체 업데이트 종료 (소요시간 : %ds)... ***" % ((end_time - start_time)/3600))






# ## Bert Finetuning when ndate%8 == 0
#     if (nday+1)%8 != True:
#         bft = BertFineTuning()
#         bft.train()
## 지금 mklabel 에서 withlabel 에 해당하는 코드가 없음 이거 만들어야 함
## fine_tuning 폴더에 저장해뒀다가 한꺼번에 combine하고 나서 그거를 싹 지우는 방식으로 가면 될듯 ( 어차피 past_combined랑 합쳐져서 다 저장 됨 )


# TODO
# stopwords 하나로 사용할 거 정하기 제한 사항 있는지 확인
# Unique_O 지금 방식(메모리최소화)으로는 사용이 어려워서 안함 ( Load Data 일때 불러오는 방식이어서 안됨 - 지우게 되면 사이즈가 계속 바껴서 )
# 그래서 지워버림 일단
# prackchi 도 마찬가지로, 현재 s를 속여서 ntn을 학습하고 있음
# 이 때, s를 random으로 추출 시, 동일 한 값이 들어가게 된다면 svo가 모두 같아서 label만 다르게 됨 -> 학습에 큰 방해
# 이 부분 또한 해결해야함