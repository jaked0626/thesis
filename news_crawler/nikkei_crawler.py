# nikkei_crawler.py: Crawls nikkei shinbun database for articles with the following search terms:
# 新型　OR 感染
# in their headlines. 
# To use, open https://waseda-jp.libguides.com/az.php?q=%E6%97%A5%E6%9C%AC%E7%B5%8C%E6%B8%88%E6%96%B0%E8%81%9E&p=1, 
# login using WasedaID, and configure search parameters. I searched for Articles in Nikkei shinbun Morning print
# with topic keywords 新型 and 感染　between 2020-01-01 and 2021-12-31

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
import re

@dataclass
class NikkeiDriver:

    def __init__(self):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        
    def __del__(self):
        self.driver.close()

    def login(self):
        # go to waseda database page 
        self.driver.get("https://waseda-jp.libguides.com/az.php?q=%E6%97%A5%E6%9C%AC%E7%B5%8C%E6%B8%88%E6%96%B0%E8%81%9E&p=1")

        # click on nikkei telecom 21
        nikkei_link = self.find_element(By.PARTIAL_LINK_TEXT, "日経テレコン21")
        self.click(nikkei_link)

        # login to waseda account
        print("Please login to your Waseda account")
        sleep(25)

        # switch to nikkei tab
        original_handle = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != original_handle:
                self.driver.switch_to.window(handle) # nikkei tab


    def configure_search(self):
        print("configure search parameters")
    
    def crawl_search_results(self):
        dates = []
        titles = []
        articles = []
        go = True
        while go:
            # select all articles shown in page
            select_all_btn = self.driver.find_element("xpath", '//div[@class="btnAreaHorizontal"]/a[2]')
            self.click(select_all_btn)
            sleep(3)

            # open all articles
            open_all_btn = self.driver.find_element("xpath", '//div[@class="btnAreaHorizontal"]/a[1]')
            self.click(open_all_btn)
            sleep(10)

            # get html and build soup 
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # parse soup to find all title blocks
            title_blocks = soup.find_all("div", class_="title")

            # extract dates from title block
            dates += [title_block.text.split()[-5] for title_block in title_blocks]
            with open("dates.txt", "w") as f:
                f.writelines("\n".join(dates))

            # extract title from title block
            titles += [title_block.find('b').text for title_block in title_blocks]
            with open("titles.txt", "w") as f:
                f.writelines("\n".join(titles))

            # parse soup to find all articles
            article_blocks = soup.find_all("div", class_="article")
            articles += [article_block.text for article_block in article_blocks]
            with open("articles.txt", "w") as f:
                f.writelines("\n".join(articles))

            self.driver.back()
            
            try:
                btn = self.driver.find_element(By.PARTIAL_LINK_TEXT, "次へ")
                self.click(btn)
                sleep(10)
            except Exception:
                go = False
        
        mainichi_df = pd.DataFrame({"date": dates, "title": titles, "text": articles})
        mainichi_df["date"] = pd.to_datetime(mainichi_df["date"], format="%Y.%m.%d")
        mainichi_df.to_csv("./data/news/mainichi.csv", index=False)
        





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


