import pandas as pd
import numpy as np
import os
### 0815 오빠가 아래 세개는 안쓰이는데도 남겨뒀었거든 삭제해도 될 것 같지?
#import sys
#import requests
#import re
from datetime import date, datetime
from preprocesser import Preprocesser


###그냥 preprocesser에 다 넣어도 될 것 같음.. ! 
###complete_data 

class Secondarybattery_preprocesser(Preprocesser):
    


    # 피쳐 데이터 컬럼 이름 설정(2차전지) 
    def feature_col_name(self, df, feature_name):

        #0815 ol_list 도 따로 txt파일로 받을 수 있도록 해야함.

        #키와 딕셔너리로 받도록
        col_lists= {'dubai':['Date', 'dubai_price','dubai_price_change','dubai_price_change_rate','dubai_stock'] ,\
         'nickel':['Date', 'nickel_price','nickel_price_change','nickel_price_change_rate','nikcel_stock'] ,\
         'mineral': ['Date', 'mineral_index','mineral_index_change','mineral_index_change_rate' ,'major_mineral_index','major_mineral_change','major_mineral_change_rate'] ,\
         'wti': ['Date', 'wti_price','wti_price_change','wti_price_change_rate','wti_stock'] ,\
         'lithum': ['Date', 'lithum_price','lithum_price_change','lithum_price_change_rate','lithum_stock'] ,\
         'cobalt_l':  ['Date', 'cobalt_l_price','coblat_l_price_change','cobalt_l_price_change_rate','cobalt_l_stock'] ,\
         'cobalt_r':  ['Date', 'cobalt_r_price','coblat_r_price_change','cobalt_r_price_change_rate','cobalt_r_stock'] ,\
         'brent': ['Date', 'brent_price','brent_price_change','brent_price_change_rate','brent_stock'] ,\
         'kospikosdaq':  ['Date', 'kosdaq','kospi'] ,\
         'USD_KRW':['Date','USD_KRW_close','USD_KRW_open','USD_KRW_high','USD_KRW_low','USD_KRW_change_rate'],\
         'South Korea 5-Year Bond Yield Historical':['Date','5-Year Bond Yield_close','5-Year Bond Yield_open','5-Year Bond Yield_high','5-Year Bond Yield_low','5-Year Bond Yield_change_rate'] ,\
         'kospi_volatility':['Date','kospi_volatility_close','kospi_volatility_open','kospi_volatility_high','kospi_volatility_low', 'kospi_volatility_volume','kospi_volatility_change_rate']}

       # for name in feature_name:
        col_list = col_lists[feature_name]
        df.columns = col_list 
        return df

        ######0815 
        ### 광물종합지수라는 파일 안에 1,2 sheet 모두 사용해야함. complete_data돌리기 전에 
        #   sheet1, sheet2를 merge해 광물종합,메이저금속지수.csv로 저장해야함.
        #   아니면 14줄 아래 쯤 list에서 해당 파일 빼고 돌려 합산한 이후에 광물종합지수 안의 두 시트를 합산하는 별도의 과정 필요. ( 그 코드는 안 넣음 )
        
        ##마니오빠 코드 

    """
mineral_index_1 = pd.read_excel("{}".format(mineral_index_name), thousands=',', sheet_name='광물종합지수')
mineral_index_2 = pd.read_excel("{}".format(mineral_index_name), thousands=',', sheet_name='메이저금속지수')
mineral_index = pd.merge(mineral_index_1,mineral_index_2, on = '기준일',how='outer')
    
    """
    
    

    def complete_data(self, industry, company):
        feature_data = pd.DataFrame()

        def complete_feature_data(industry, feature, feature_name, start_date):  
            data = pd.read_excel(f"G:\\공유 드라이브\\Boad ADV Stock\\{industry}\\feature_data\\{feature}.xlsx", thousands=',')
            data = self.feature_col_name(data, feature_name)
            data = self.replace_date_form(data)
            data = self.set_feature_data(data, start_date)
            data.replace('-',np.nan, inplace=True)
            data.sort_values(by='Date', ascending=True, inplace=True)
            feature_data = pd.concat([feature_data, data])
             ### 0815 def complete_data 두번째 줄에서 빈 데이터프레임으로 feature_data를 정의했다고 생각했는데 인식못함. 도와주세요... 

            #0815. ver2
            #feature_data = pd.merge(feature_data, data, on='Date', how='left' )
            ### 0815 근데 시작 날짜가 다 달라서 위에껄루 해야하지 않을까 싶음. > 오류뜸ㅎㅎ
            
            ### 만약에 위에껄루 한다면 Date 행이 필요하니까 빈 df면 안될 것 같고 예를들어 feature_data = dubai로 초기화하고 
            ### merge시켜야할 것 같은데 그렇다면 complete_feature_data에서 merge문장을 빼고 10줄 포문 아래에 [1:]로 인덱싱한 포문에 merge문장 넣어.. 합친다?
            ### >>>>>>>>> 뭔가 조잡해보여서 일단 패스.. ^^ 
            ### ver1으로 잘.. 해결해보아요.. ...ㅎㅎㅎ
        
            # feature_data = pd.merge(feature_data, data, on='Date', how='left' )
            # 0815 파일마다 시작날짜가 달라서 위에껄루 해야하지 않을까 싶음.
           
        
             
            return feature_data
        

        ##아래 list들은 따로 txt파일로 받아야함.
        industries = ['2차전지', '2차전지', '2차전지', '2차전지', '2차전지', '2차전지', '2차전지', '2차전지', '2차전지', '2차전지', \
                '2차전지', '2차전지']
        features = ['니켈가격', '두바이유가격(주간)','WTI유가격(주간)','브렌트유가격(주간)','코발트가격(메탈LME)','코발트가격(rotterdam)', \
                '광물종합,메이저금속지수', '탄산리튬가격','kospi,kosdaq','USD_KRW 내역','South Korea 5-Year Bond Yield Historical','kospi_volatility']
        feature_names = ['nickel','dubai','wti','brent','cobalt_l','cobalt_r','mineral','lithum','kospikosdaq','USD_KRW',\
                'South Korea 5-Year Bond Yield Historical','kospi_volatility']
        start_dates = ['2010-01-04','2010-01-08','2010-01-08','2010-01-08','2010-05-19', '2010-01-01','2011-01-04','2010-02-01',\
                '2010-01-04','2011-01-03','2010-01-04','2013-08-06']

        for industry, feature, feature_name, start_date in zip(industries, features, feature_names, start_dates):
            feature_data = self.complete_feature_data(industry, feature, feature_name, start_date)
    

        #data = fn_data 
        fn_data= pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\fn_data\{}.xlsx".format(industry,company),encoding='cp949', thousands=',')
        fn_data = self.set_col_names(fn_data)
        fn_data = self.set_funda_data(fn_data , '2010-01-04') ###0815 날짜 확인해야함 + 기업마다 다 다를 수 있음.
        fn_data.replace('-',np.nan, inplace=True)
        
        

        ##0815 technical_data는 아래와 중복되어 삭제함.
        

        #event with tech
        event_with_tech_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_with_tech_{}.xlsx".format(industry, company), thousands=',')
        event_with_tech_data.columns = ['Date', 'rsi_14', 'macd', 'cci', 'adx', 'stoch_slowk',\
             'stoch_slowd', 'willr', 'momentum', 'roc', 'ema20', 'adosc', 'obv', 'event1', 'event2', 'event3', 'event4', 'event5', 'event6',\
             'event7', 'event8', 'event9', 'event10', 'event11', 'event12',\
             'event13', 'event14', 'event15', 'event16']

        
     
        #시가 종가 profit
        tech_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\tech_data\{}_tech.xlsx".format(industry, company),thousands=','))
        tech_data = tech_data[5:]
        tech_data.columns = ['Date','open','high','low','last','volume']
        tech_data['profit_rate'] = (tech_data['last'] - tech_data['open'])/tech_data['open']*100
        ## 0815 마니오빠 파일에 주석처리되어있길래 아래 주석처리함.
        #tech_data.profit_rate = tech_data.profit_rate.shift(-1)
        tech_data.Date = pd.to_datetime(tech_data.Date)
        profit_rate = tech_data[['Date','profit_rate']]


        #data concat
        fn_data.drop('profit_rate',axis=1,inplace=True)
        df_list = [fn_data, feature_data, event_with_tech_data, profit_rate]
        for list_data in df_list:
            list_data.Date = pd.to_datetime(list_data.Date)
        data= pd.DataFrame()       
        for dataname in df_list[1:]:
            data = pd.merge(data, dataname, on ='Date',how='left')   
        data.iloc[:,2:] = data.iloc[:,2:].astype('float64')

        ##롤링 => n일마다의 이동평균선

        data['profit_rate'] = data.profit_rate.shift(periods=-1)
        data['rate_rolling_3'] = data.profit_rate.rolling(window=3).mean()
        data['rate_rolling_5'] = data.profit_rate.rolling(window=5).mean()
        data['rate_rolling_10'] = data.profit_rate.rolling(window=10).mean()
        data['rate_rolling_30'] = data.profit_rate.rolling(window=30).mean()
        data['rate_rolling_120'] = data.profit_rate.rolling(window=120).mean()
        data['rate_rolling_150'] = data.profit_rate.rolling(window=150).mean()
        
        
        os.chdir(r'G:\공유 드라이브\Boad ADV Stock\{}'.format(industry))
        data.to_csv("data.csv")
        
        
        
        ### 0815 마니오빠파일은 위까지만 저장되어있음. 일단 아래 첨부하겠음

        data.set_index('Date', inplace=True)
        data.close_kospi_volatility[~data.close_kospi_volatility.isnull()]

        return data
        
        #fill null data
        ### null plot 보고 시작일자 결정 
        ### 기업마다 달라질 수 있음. >확인하고 조건문 만들기
        
        """
        ex.축산업
        data = data['2013-08-06':]
        data.loc[:,'close_livecattle_f':'BDI'] = data.loc[:,'close_livecattle_f':'BDI'].fillna(method='ffill')
        data.loc[:,'close_leanhog_f':'change%_leanhog_f'] = data.loc[:,'close_leanhog_f':'change%_leanhog_f'].fillna(method='ffill')
        
        return data
        """
        # 실수형인 종속변수를 범주형으로 바꾸기 위한 작업
    def convert_target(self, x):
        THES = 0.2
        
        if x >= 8 * THES:
            return 2
        elif x >= THES and x < 8 * THES:
            return 1
        elif x >= (-THES) and x < THES:
            return 0
        elif x >= (-8*THES) and x < -THES:
            return -1
        else:
            return -2

        return data
    ### up down예상 
    ### regression 대신 classfication'
    # 양 극단에 있을수록 타격이 크니 학습을 더 잘되게
    ## 변화율이 더 클때 잘 맞추는게 중요함.
    ### 회사마다 다르게 설정할떄 더 잘나올지도
        

    def feature_engineering():
        pass