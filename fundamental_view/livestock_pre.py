import pandas as pd
import numpy as np
import os
from preprocesser import Preprocesser

class livestock_preprocesser(Preprocesser):


    def feature_col_name(self, df, feature_name):
        
        #0815 col_list는 따로 받아야함.
        col_lists={'livecattle_f':['Date','livecattle_f_close','livecattle_f_open','livecattle_f_high','livecattle_f_low','livecattle_f_volume','livecattle_f_change_rate'] ,\
         'bdi':['Date','bdi'] ,\
         'leanhog_f':['Date','leanhog_f_close','leanhog_f_open','leanhog_f_high','leanhog_f_low','leanhog_f_volume','leanhog_f_change_rate'],\
         'kospikosdaq': ['Date', 'kosdaq','kospi'] ,\
         'USD_KRW':['Date','USD_KRW_close','USD_KRW_open','USD_KRW_high','USD_KRW_low','USD_KRW_change_rate'], \
         'South Korea 5-Year Bond Yield':['Date','5-Year Bond Yield_close','5-Year Bond Yield_open','5-Year Bond Yield_high','5-Year Bond Yield_low','5-Year Bond Yield_change_rate'], \
         'kospi_volatility':['Date','kospi_volatility_close','kospi_volatility_open','kospi_volatility_high','kospi_volatility_low', 'kospi_volatility_volume','kospi_volatility_change_rate']}

        #for name in col_lists:
        col_list = col_lists[feature_name]
        df.columns = col_list
        return df

    def complete_data(self, industry, company):
        feature_data = pd.DataFrame()


        def complete_feature_data(industry, feature, feature_name, start_date):
            data = pd.read_csv(f"G:\\공유 드라이브\\Boad ADV Stock\\{industry}\\feature_data\\{feature}.csv",  thousands=',')
            ### 0815 아래 코드가 안되어서 위에껄루 바꿈 
            ### 0815 xlsx로 받으면 날짜 포멧이 43989 이런식으로 바뀌어서 csv파일로 받도록 설정. 
            ### csv파일로도 저장해놨음. (폴더가 매우 지저분해짐..)
            ### 0815 bdi의 경우 다른 파일들과 달리 index 행이 없음. 근데 첫번째 행이 결측값. 인덱스로 처리하고 인덱스 이름 바꿀테니 따로 코드를 작성하지는 않음.
            #data = pd.read_excel(f"G:\\공유 드라이브\\Boad ADV Stock\\f'{industry}'\\feature_data\\f'{feature}''.xlsx', thousands=','")
            data = self.feature_col_name(data, feature_name)
            data = self.replace_date_form(data)
            data = self.set_feature_data(data, start_date)
            data.replace('-',np.nan, inplace=True)
            data.sort_values(by='Date', ascending=True, inplace=True)
            #0815. ver1
            feature_data = pd.concat([feature_data, data]) 
             ### 0815 def complete_data 두번째 줄에서 빈 데이터프레임으로 feature_data를 정의했다고 생각했는데 인식못함. 도와주세요... 

            #0815. ver2
            #feature_data = pd.merge(feature_data, data, on='Date', how='left' )
            ### 0815 근데 시작 날짜가 다 달라서 위에껄루 해야하지 않을까 싶음. > 오류뜸ㅎㅎ
            
            ### 만약에 위에껄루 한다면 Date 행이 필요하니까 빈 df면 안될 것 같고 예를들어 feature_data = livecattle_f 초기화하고 
            ### merge시켜야할 것 같은데 그렇다면 complete_feature_data에서 merge문장을 빼고 10줄 포문 아래에 [1:]로 인덱싱한 포문에 merge문장 넣어.. 합친다?
            ### >>>>>>>>> 뭔가 조잡해보여서 일단 패스.. ^^ 
            ### ver1으로 잘.. 해결해보아요.. ...ㅎㅎㅎ
        
            # feature_data = pd.merge(feature_data, data, on='Date', how='left' )
            # 0815 파일마다 시작날짜가 달라서 위에껄루 해야하지 않을까 싶음.
           
            
            return feature_data

        ##0815 아래 list들은 따로 txt파일로 받아야함.
        industries = ['축산업', '축산업', '축산업', '축산업', '축산업', '축산업', '축산업']
        features = ['livecattle_f', 'bdi', 'leanhog_f', 'kospi,kosdaq', 'USD_KRW', 'South Korea 5-Year Bond Yield', 'kospi_volatility']
        feature_names = ['livecattle_f', 'bdi', 'leanhog_f', 'kospikosdaq', 'USD_KRW', 'South Korea 5-Year Bond Yield', 'kospi_volatility']
        ##0815 start_data확인해야함. (아마맞을텐데.. 왜 이렇게 써놨지..)
        start_dates = ['2010-01-01','2011-01-03','2010-01-01','2010-01-04', '2010-01-01','2010-01-04','2013-08-06']

        for industry, feature, feature_name, start_date in zip(industries, features, feature_names, start_dates):
            feature_data = complete_feature_data(industry, feature, feature_name, start_date)


        #fn_data 
        fn_data= pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\fn_data\{}.xlsx".format(industry,company),encoding='cp949', thousands=',')
        fn_data = self.set_fn_data_col_names(fn_data)
        fn_data = self.set_funda_data(fn_data , '2009-01-02') ###0815 날짜 확인해야함 + 기업마다 다 다를 수 있음.
        fn_data.replace('-',np.nan, inplace=True)


        ##0815 technical_data는 아래와 중복되어 삭제함.
  
        ##event with tech
        event_with_tech_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_with_tech_{}.xlsx".format(industry, company), thousands=',')
        event_with_tech_data.columns = ['Date', 'rsi_14', 'macd', 'cci', 'adx', 'stoch_slowk',\
       'stoch_slowd', 'willr', 'momentum', 'roc', 'ema20', 'adosc', 'obv', 'event1', 'event2', 'event3', 'event4', 'event5', 'event6',\
             'event7', 'event8', 'event9', 'event10', 'event11', 'event12',\
             'event13', 'event14', 'event15', 'event16']
    

        #시가 종가 profit
        tech_data = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\tech_data\{}_tech.xlsx".format(industry, company),thousands=',')
        tech_data = tech_data[5:]
        tech_data.columns = ['Date','open','high','low','last','volume']
        tech_data['profit_rate'] = (tech_data['last'] - tech_data['open'])/tech_data['open']*100
        ###0815 아랫줄 의미를 잘 모르겠음. 이지바이오 파일에는 주석처리 안하고 2차전지에는 주석처리되어있어서 일단 둡니다.
        tech_data.profit_rate = tech_data.profit_rate.shift(-1)
        tech_data.Date = pd.to_datetime(tech_data.Date)
        profit_rate = tech_data[['Date','profit_rate']]

        #data concat
        fn_data.drop('profit_rate',axis=1,inplace=True)
        df_list = [fn_data, feature_data, event_with_tech_data, profit_rate]
        for list_data in df_list:
            list_data.Date = pd.to_datetime(list_data.Date)
        data = fn_data  
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

        ### 0815 정만오빠 파일에는 여기까지 작성되어있음. 일단 아래 코드도 남겨놓겠음.


        data.set_index('Date', inplace=True)
        data.close_kospi_volatility[~data.close_kospi_volatility.isnull()]

        return data

        #fill null data
        ### null plot 보고 시작일자 결정 
        ### 기업마다 달라질 수 있음. > 확인하고 조건문 만들기
        
        # 아래는 이지바이오.
"""
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
        
    ### up down예상 
    ### regression 대신 classfication'
    # 양 극단에 있을수록 타격이 크니 학습을 더 잘되게
    ## 변화율이 더 클때 잘 맞추는게 중요함.
    ### 회사마다 다르게 설정할떄 더 잘나올지도

        return data

    def feature_engineering():
        pass