from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.parse import quote
import time, random, re
import pandas as pd
import os

#네이버 뉴스 수집 함수
local_path = r'C:\Users\JSH\Desktop\Github\stock_prediction\\'
db_path = r'G:\공유 드라이브\Boad ADV Stock\\'
def get_naver_news(path, query, yyyy, mm, dd):
    # 검색어 인코딩
    naver_query = quote((query.encode('utf-8')))

    # 제목에 검색어 포함인 경우(field=1)
    # url = "https://search.naver.com/search.naver?where=news&query=" + naver_query + "&sm=tab_opt&sort=0&photo=0&field=1&reporter_article=&pd=3&ds=" + yyyy + "." + mm + "." + dd + "&de=" + yyyy + "." + mm + "." + dd + "&docid=&nso=so%3Ar%2Cp%3Afrom" + yyyy + mm + dd + "to" + yyyy + mm + dd + "2Ca%3Aall&mynews=0&mson=1&refresh_start=0&related=0"

    # 본문에 검색어 포함인 경우(field=0)
    url = "https://search.naver.com/search.naver?where=news&query=" + naver_query + "&sm=tab_opt&sort=0&photo=0&field=0&reporter_article=&pd=3&ds=" + yyyy + "." + mm + "." + dd + "&de=" + yyyy + "." + mm + "." + dd + "&docid=&nso=so%3Ar%2Cp%3Afrom" + yyyy + mm + dd + "to" + yyyy + mm + dd + "2Ca%3Aall&mynews=0&mson=1&refresh_start=0&related=0"

    naver_news_headers = []
    naver_news_dates = []
    naver_news_links = []
    naver_news_press = []

    # 구글 가상 브라우징 옵션, 헤더 정보 추가
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('disable-gpu')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.146 Safari/537.36")

    # 구글 드라이버 경로 설정, 옵션 및 헤더 적용
    driver = webdriver.Chrome(r'C:\Users\JSH\Desktop\Github\stock_prediction\chromedriver.exe', chrome_options=options)
    driver.execute_script("Object.defineProperty(navigator, 'plugins', {get: function() {return[1, 2, 3, 4, 5];},});")
    driver.execute_script("Object.defineProperty(navigator, 'languages', {get: function() {return ['ko-KR', 'ko']}})")

    # 구글 드라이버 가상 브라우징 및 페이지 스크랩
    naver_virtual_url = url
    driver.get(naver_virtual_url)
    driver.implicitly_wait(3)  # 로딩 완료까지 대기
    naver_page = driver.page_source.encode('utf-8')

    print("*** 수집 대상 페이지 탐색 시작 ***")
    print(naver_virtual_url)

    # 웹페이지 파싱
    naver_soups = BeautifulSoup(naver_page, 'html.parser')

    i = 1

    while True:
        try:
            for naver_news in naver_soups.find_all("a", {"class": "_sp_each_title"}):
                naver_news_headers.append(naver_news.attrs['title'])
                n_raw_date = naver_news.find_next("dd", {"class": "txt_inline"}).get_text()
                naver_news_dates.append(re.findall(r"\b[0-9]+.[0-9]+.[0-9]+\b", n_raw_date)[0])
                naver_news_links.append(naver_news.find_next("a").attrs['href'])

            for naver_news in naver_soups.find_all("span", {"class": "_sp_each_source"}):
                naver_news_press.append(naver_news.get_text())
            
            element = driver.find_element_by_partial_link_text("다음페이지")
            element.click()

            print("{} 번째 수집 및 페이지 이동중".format(i))
            driver.implicitly_wait(3)
            naver_page = driver.page_source.encode('utf-8')
            naver_soups = BeautifulSoup(naver_page, 'html.parser')
            i = i+1

        except:
            print("마지막 {} 번째 페이지 이동 및 크롤링 완료\n".format(i))
            break

    driver.quit()

    naver_news_df = pd.DataFrame([naver_news_headers, naver_news_dates, naver_news_links, naver_news_press]).T
    naver_news_df.columns = ['header', 'date', 'link', 'press']

    # 링크의 경우 저장 시 오류가 발생, ExcelWriter로 저장
    writer = pd.ExcelWriter(path + query + '_naver_news_' + str(yyyy + mm + dd) + '.xlsx',
                            engine='xlsxwriter',
                            options={'strings_to_urls': False})
    naver_news_df.to_excel(writer)
    writer.close()

    print("*** 수집 대상 저장 완료 ***")
    print("------------------------------------------------------\n")

    # 웹 페이지 탐색 불규칙화
    time.sleep(random.uniform(1, 3))

#combine 함수
def news_combine(industry, keyword):
    os.chdir(r"C:\Users\JSH\Desktop\Github\stock_prediction\new\{}\{}".format(industry, keyword))
    filelist = os.listdir()
    combine = pd.DataFrame()
    for filename in filelist:
        new = pd.read_excel(filename, index_col=False)
        combine = pd.concat([combine, new])
    combine = combine.drop('Unnamed: 0', axis=1)
    combine.to_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\crawled_data\{}_합산.xlsx".format(industry, keyword), index=0)

#new files combine for preprocess
def update_news_preprocess(industry, keyword):
    os.chdir(local_path + r"new\{}\{}\\".format(industry, keyword))
    filelist = os.listdir()
    combine = pd.DataFrame()
    for filename in filelist:
        new = pd.read_excel(filename, index_col=False)
        combine = pd.concat([combine, new])
    combine = combine.drop('Unnamed: 0', axis=1)
    combine.to_excel(local_path + r"new\{}\합산\{}_합산.xlsx".format(industry, keyword), index=0)
    
    return combine


