import pandas as pd
import numpy as np
import re
from openie import StanfordOpenIE
from behavior_view.fine_tuning.bert_fine_tuning import BertFineTuning
from pypapago import Translator
import argparse
from ast import literal_eval
from tqdm import tqdm
import random
import json

##fine tuning을 해야 안에 있는 단어들의 vector를 가질 수 있삼

class Text_preprocess:
    def __init__(self):
        # 어떤 임베딩 모델을 불러올지에 따라서 다름
        self.translator = Translator()
        self.bft = BertFineTuning()

    def word_embbeding(self, text):
        return self.bft.word_embedding(text)

    def papago(self, text, source = "ko", target = "en"):
        try:
            result = self.translator.translate(text, source = source, target = target)
        except:
            result = ""

        return result

    def SVO_extractor(self, data):
        with StanfordOpenIE() as client:
            svo_pos = {'s_pos' : [], 'v_pos' : [], 'o_pos': [], 'label' : [], 'date' : []}
            for index, row in tqdm(data.iterrows()):
                try:
                    for sentence in client.annotate(row['header']):    
                        svo_pos['s_pos'].append(sentence['subject'])
                        svo_pos['v_pos'].append(sentence['relation'])
                        svo_pos['o_pos'].append(sentence['object'])
                        svo_pos['label'].append(1)
                        svo_pos['date'].append(row['date'])
                except AttributeError:
                    pass

        return svo_pos

    #괄호 지우기 + 필요한 한자 한글로 변환하기


    def svo_embedding(self, svo_pos):
        for i in tqdm(range(len(svo_pos['s_pos']))):
            svo_pos['s_pos'][i] = self.word_embbeding(svo_pos['s_pos'][i])
            svo_pos['v_pos'][i] = self.word_embbeding(svo_pos['v_pos'][i])
            svo_pos['o_pos'][i] = self.word_embbeding(svo_pos['o_pos'][i])

        return svo_pos

    def clean_text(self, text):
        patterns = [r"\([^<|>]*\)",
                    r"\[[^<|>]*\]",
                r"\<[^<|>]*\>",
                r"[^\w\s]"]
        for p in patterns:
            try:
                text = re.sub(p, '',text)
            except TypeError:
                text = ''

        for hanza in replace_dict.keys():
            if hanza in text:
                text = re.sub(hanza, replace_dict[hanza], text)

        return text.strip()

    #단어 지우기
    
    def mkstopwords(self, stopword):
        stopwords = ''
        for text in stopword:
            stopwords += '|' + text

        return stopwords[1:]

    #총 길이가 j가 안되는 부분 전부 지우기
    #남은 한자 있는 열 날리기 ( 무슨 의미인지 예측 불가이므로 그냥 날림 )
    def mk_del_smallwords(self, data, j):
        data_index=[]
        for i in range(len(data)):
            if len(re.split(' ', str(data.iloc[i]['header']))) <= j:
                data_index.append(data.iloc[i].name)

        for i in range(len(data)):
            if re.search(r"[\u4e00-\u9fff]", str(data.iloc[i]['header'])):
                data_index.append(data.iloc[i].name)
        
        data = data.drop(data_index)

        return data

    def stop_words(self, data, stopword, i):
        stopwords = self.mkstopwords(stopword)
        data = data.loc[~data['header'].str.contains(stopwords, na=False)]
        
        data =self.mk_del_smallwords(data, i)
        
        return data


if __name__ == "__main__":

    txt = Text_preprocess()
    parser = argparse.ArgumentParser()
    parser.add_argument('stopwordstxt', help= 'stopwords txt file')
    parser.add_argument('replacedicttxt', help= 'replace dict txt file')
    parser.add_argument('--i', required=True, type= str,  help= '산업 이름')
    parser.add_argument('--c', required=True, type= str,  help= '기업 이름')
    args = parser.parse_args()
    with open(args.stopwordstxt, encoding='utf-8') as file1:
        stopwords = file1.read().splitlines()
    with open(args.replacedicttxt, encoding='utf-8') as file2:
        global replace_dict ## Class 안에 넣어놨어야 하는데 실슈~
        replace_dict = file2.read()
        replace_dict = literal_eval(replace_dict)
    industry = args.i
    keyword = args.c

    news_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\crawled_data\{}_합산.xlsx".format(industry, keyword))
    news_data = txt.stop_words(news_data, stopwords, 3)
    news_data.header = news_data.header.apply(txt.clean_text)
    news_data.to_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\crawled_data\정제\{}_정제.xlsx".format(industry, keyword))
    news_data.header = news_data.header.apply(txt.papago)
    news_data = news_data[news_data['header'] != ""]

    news_data.to_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\crawled_data\{}_translate.xlsx".format(industry, keyword))

    news_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\crawled_data\{}_translate.xlsx".format(industry, keyword))
    # news_data = news_data.drop(['Unnamed: 0'], axis=1)

    svo_pos, svo_neg = txt.SVO_extractor(news_data)
    with open(r"G:\공유 드라이브\Boad ADV Stock\{}\svo_pos_{}.json".format(industry, keyword), mode='w', encoding='utf-8') as f:
        json.dump(svo_pos, f)
    with open(r"G:\공유 드라이브\Boad ADV Stock\{}\svo_neg_{}.json".format(industry, keyword), mode='w', encoding='utf-8') as f:
        json.dump(svo_neg, f)


