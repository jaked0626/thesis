# yomiuri_crawler.py: Crawls yomiuri shinbun database for articles with the following search terms:
# 新型　OR 感染
# in their headlines. 
# To use, open https://waseda-jp.libguides.com/az.php?q=%E8%AA%AD%E5%A3%B2&p=1, 
# login using WasedaID, and configure search parameters. I searched for Articles in yomiuri shinbun Morning print
# with topic keywords  新型コロナ AND 国内 AND 感染　between 2020-01-01 and 2021-12-31

from time import sleep
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pandas as pd

@dataclass
class YomiuriDriver:

    def __init__(self):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        
    def __del__(self):
        self.driver.close()

    def login(self):
        # go to waseda database page 
        self.driver.get("https://waseda-jp.libguides.com/az.php?q=%E8%AA%AD%E5%A3%B2&p=1")

        # click on yomiuri yomidasu
        yomiuri_link = self.find_element(By.PARTIAL_LINK_TEXT, "ヨミダス歴史")
        self.click(yomiuri_link)

        # login to waseda account
        input("Login to your Waseda Account and press <ENTER> to continue: ")

        # switch to yomiuri tab
        original_handle = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != original_handle:
                self.driver.switch_to.window(handle) # yomiuri tab


    def configure_search(self):
        input("Configure search parameters and press <ENTER> to continue: ")
        
    
    def crawl_search_results(self):
        print("crawling (this may take a while")
        # store results
        dates = []
        titles = []
        articles = []

        # initialize cache
        with open("titles.txt", "w") as f:
            f.writelines("")
        with open("dates.txt", "w") as f:
            f.writelines("")
        with open("articles.txt", "w") as f:
            f.writelines("")

        # switch to article frame 
        frame = self.driver.find_element("xpath", '//frame[@name="frame_bun"]')
        self.driver.switch_to.frame(frame)
        
        # iterate while go 
        go = True
        while go:

            # select all articles shown in page
            select_all_btn = self.driver.find_element(By.PARTIAL_LINK_TEXT, '一括表示すべて選択')
            self.click(select_all_btn)
            sleep(1)

            # open all articles
            open_all_btn = self.driver.find_element(By.LINK_TEXT, '一括表示')
            self.click(open_all_btn)
            sleep(15)

            # get html and build soup 
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # parse soup to find all issue info # width:60px;vertical-align:middle;padding-left:7px;padding-right:3px;
            issue_blocks = soup.find_all("th", style="width:60px;vertical-align:middle;padding-left:7px;padding-right:3px;")
            issues = [issue_block.text for issue_block in issue_blocks]

            # parse soup to find all page number info th style="width:38px;vertical-align:middle;padding-left:7px;padding-right:3px;"
            page_num_blocks = soup.find_all("th", style="width:38px;vertical-align:middle;padding-left:7px;padding-right:3px;")
            page_nums = [page_num.text for page_num in page_num_blocks if page_num.text[-1] == "頁"]


            # parse soup to find all article titles on first page
            title_blocks = soup.find_all("th", class_="wp50")
            titles_temp = [title_block.text for title_block in title_blocks]
            titles_temp = [title for i, title in enumerate(titles_temp) if (issues[i] == "東京朝刊" and page_nums[i] == "01頁")]

            with open("titles.txt", "a") as f:
                f.writelines("\n".join(titles_temp) + "\n")

            # parse soup to find all article dates
            date_blocks = soup.find_all("th", style="width:80px;vertical-align:middle;")
            dates_temp = [date_block.text for date_block in date_blocks]
            dates_temp = [date for i, date in enumerate(dates_temp) if (issues[i] == "東京朝刊" and page_nums[i] == "01頁")]
            with open("dates.txt", "a") as f:
                f.writelines("\n".join(dates_temp) + "\n")

            # parse soup to find all articles
            article_blocks = soup.find_all("p", class_="mb10")
            articles_temp = [article_block.text.replace(",", "").replace("\n","").replace('”', '') for article_block in article_blocks]
            articles_temp = [article for i, article in enumerate(articles_temp) if (issues[i] == "東京朝刊" and page_nums[i] == "01頁")]
            with open("articles.txt", "a") as f:
                f.writelines("\n".join(articles_temp) + "\n")

            # add temporary titles, articles, dates
            dates += dates_temp
            titles += titles_temp
            articles += articles_temp

            self.driver.back()
            sleep(5)
            
            try:
                # switch to article frame 
                frame = self.driver.find_element("xpath", '//frame[@name="frame_bun"]')
                self.driver.switch_to.frame(frame)
                # next page
                btn = self.driver.find_element(By.PARTIAL_LINK_TEXT, "次のページ")
                self.click(btn)
                sleep(10)
            except Exception:
                go = False
        
        yomiuri_df = pd.DataFrame({"date": dates, "title": titles, "text": articles})
        yomiuri_df["date"] = pd.to_datetime(yomiuri_df["date"], format="%Y.%m.%d")
        yomiuri_df.to_csv("./data/news/yomiuri.csv", index=False)
        

    def click(self, element):
        """ click unclickable elements on page """
        self.driver.execute_script('arguments[0].click();', element)

    def find_element(self, method: By, path: str):
        """ 
        wait for appearance of element and find it 
        Inputs
            method (By): search method 
            path (str): xpath, link, id, etc. Search path.  
        Returns
            element: web element (if not found, returns False)
        """
        timeout = 20
        try:
            element_present = EC.presence_of_element_located((method, path))
            element = WebDriverWait(self.driver, timeout).until(element_present)
            return element
        except TimeoutException:
            # add refresh and repeat? 
            print("Timed out waiting for element with path {} to load".format(path))
            raise Exception("Timed out waiting for element with path {} to load".format(path))


    def main(self):
        self.login()
        self.configure_search()
        self.crawl_search_results()


def main():
    y = YomiuriDriver()
    y.main()

if __name__=="__main__":
    main()
