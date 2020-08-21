import pandas as pd
import numpy as np
from datetime import datetime

class Preprocesser:
    def __init__(self):
        pass 
    
    ##0815 함수 이름 수정함.
    # 재무 데이터 컬럼 이름 설정
    def set_fn_data_col_names(self, df):
        col_list = ['Date', 'asset(1000)', 'debt(1000)', 'sales(1000)',
                    'adjusted_price', 'profit_rate', 'volumne(mean_5d)(week)',
                    'volumne(week)','profit(1000)',
                    'term_profit(1000)', 'gross_margin(1000)','adjusted_BPS', 'adjusted_EPS']
        df.columns = col_list
        return df
    
    def add_feature_name(self, feature_name_list, file_name):
        dotpos = file_name.find('.')
        file_name = file_name[:dotpos]
        new_col = ['Date']
        for i in range(feature_name_list)-1:
            new_name = file_name+feature_name_list[i+1]
            new_col += new_name
        return new_col

    ###0815 함수내부에서 RESET_INDEX까지 시키도록 수정함.

    # 인덱스 설정하고 원하는 연도부터 프레임 설정(재무)
    def set_funda_data(self, data, start_date):

        """
        set_funda_data(df,'2010-01-05')
        """
        data = data.iloc[5:]

        data.Date =  pd.to_datetime(data.Date)
        data.set_index(['Date'],inplace=True)
        data = data.loc[start_date:]
        data = data.reset_index()
        return data

    
    ###0815 함수내부에서 RESET_INDEX까지 시키도록 수정함.

    # 인덱스 설정하고 원하는 연도부터 프레임 설정(피쳐)
    def set_feature_data(self, data, start_date):
        
        data.Date =  pd.to_datetime(data.Date)
        data.set_index(['Date'],inplace=True)
        data = data.loc[start_date:]
        data = data.reset_index()
        return data

    ##0815 정만오빠 파일이랑 살짝 다름. 두 함수를 합침
    
    def replace_date_form(self,data):
        data['Date'] = data['Date'].str.replace('년 ','-')
        data['Date'] = data['Date'].str.replace('월 ','-')
        data['Date'] = data['Date'].str.replace('일','')
        
        for i in data.columns[1:]:
            #print(i)
            data[i]
            col_name = '{}'.format(i)
            data[i] = data[i].astype('str')
            data[i] = data[i].str.replace('K','')
            data[i] = data[i].str.replace('%','')

        data.replace('-', np.nan, inplace=True)
        data.sort_values(by='Date', ascending=True, inplace=True)

        return data

    
    
    
    #아래는 쓰이지 않은 것들.
    
    """
    
     #화장품 fundamental data 내 존재하는 monthly data를 날짜별로 만들기  ( 현재 datetime으로 되어 있고, 모든 날짜에 값 부여 필요 )
    def mkdatetime_mth(self, data):
        data2 = data[data['년월']['년월'].str.contains("월")]
        index = data2.index
        for i, mth in enumerate(data2['년월']['년월']):
            m = int(mth[1:-1])
            idx = index[i]
            idx = idx - m
            y = int(data['년월']['년월'][idx][:-1])
            print(y, m)
            data2.iat[i, 0] = datetime(y, m, 1)
        return data2
    # #데이터 불러와서 datetime으로 만들고, 각 컬럼 모두 합치기, 수연이한테 각 컬럼명이 무엇을 뜻하는 지 물어볼 것
    # def read_mth_data(self, data, domain):
    #     domain = "화장품"
    #     os.chdir(r"G:\공유 드라이브\Boad ADV Stock\{}\fundamental".format(domain))
        
    #     data = pd.read_excel("330112_오렌지유.xls", skiprows = [0, 1], header=[0,1])
    #     pd.read_excel("330113_레몬유.xls", skiprows = [0, 1], header=[0,1])
    # 특정 연도 사이 휴일 리스트
    def check_holiday(self, start_year, last_year):
        
        """
       # check_hodiday(2010,2020)
        """
        total_holiday = list()
        for year in range(start_year, last_year):
            hlist = list()
            krholiday = [d.strftime('%Y-%m-%d') for d in pytimekr.holidays(year)]
            week_sat = [d.strftime('%Y-%m-%d') for d in pd.date_range('{}-01-01'.format(year),'{}-12-31'.format(year),freq='W-SAT')]
            week_sun = [d.strftime('%Y-%m-%d') for d in pd.date_range('{}-01-01'.format(year),'{}-12-31'.format(year),freq='W-SUN')]
            hlist.extend(krholiday)
            hlist.extend(week_sat)
            hlist.extend(week_sun)
            total_holiday.extend(pd.unique(hlist))
        sorted(total_holiday)
        return total_holiday
    # 특정 연도 사이 휴일 x 리스트
    def check_workday(self, start_year, last_year):
        """
      #  check_workday(2010,2020)
        """
        workday = list()
        for d in pd.date_range('{}-01-01'.format(start_year),'{}-12-31'.format(last_year)):
            if d.strftime('%Y-%m-%d') not in check_holiday(start_year, last_year):
                workday.append(d.strftime('%Y-%m-%d'))       
        return workday
    # 데이터 프레임에서 휴일 아닌 날만 뽑기
    def set_index_workday(self, df,start_year,last_year):
        work = pd.DataFrame(check_workday(start_year, last_year),columns=['Date'])
        work.Date =  pd.to_datetime(work.Date)
        df.reset_index(inplace=True)
        work_df = pd.merge(df,work, on ='Date',how='inner')
        work_df.set_index(['Date'],inplace=True)
        # work_df = pd.concat([df,work], join_axes=['Date'],join='inner',axis=0)
        return work_df
    # 회시별 재무 데이터 테이블
    def individual_feauture(self, stockname):
        df = load_funda_data(stockname)
        df = set_col_names(df)
        df = set_funda_data(df)
        return df
    # 산업 공통 피쳐
    def common_feature(self,featurename):
        
        df = load_funda_data(featurename)
        df = feature_col_name(df)
        df = set_feature_data(df)
        return df
    # 산업 공통 피쳐 테이블
    def common_feature_table(self):
        os.chdir(r'G:\공유 드라이브\Boad ADV Stock\2차전지\funda_data')
        filelist = os.listdir(os.getcwd())
        # filelist
        feature_df = pd.DataFrame()
        for feature in filelist:
            add_df = common_feature(feature)
            feature_df = pd.merge(feature_df, add_df, on='Date', how ='left')
        return
    """