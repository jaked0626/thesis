# asahi_crawler.py: Crawls asahi shinbun database for articles with the following search terms:
# 新型コロナ OR 新型肺炎 OR コロナ OR COVID  OR  (武漢 AND 肺炎)
# in their headlines. 
# To use, open https://xsearch-asahi-com.waseda.idm.oclc.org/kiji/, login using WasedaID, and
# configure search parameters. I searched for Articles in asahi shinbun Tokyo Morning print first page
# with the keywords in the headline from 2020-01-01:2021-12-31

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
class AsahiDriver:

    def __init__(self):
        self.driver = webdriver.Chrome(ChromeDriverManager().install())
        
    def __del__(self):
        self.driver.close()

    def login(self):
        self.driver.get("https://waseda-jp.libguides.com/az.php?q=%E6%9C%9D%E6%97%A5&p=1")
        print("Please login to your Waseda account")
        sleep(15)
        asahi_link = self.find_element(By.PARTIAL_LINK_TEXT, "朝日新聞クロスサーチ")
        self.click(asahi_link)
        original_handle = self.driver.current_window_handle
        # switch to asahi tab
        asahi_handle = ""
        for handle in self.driver.window_handles:
            if handle != original_handle:
                asahi_handle = handle
        self.driver.switch_to.window(asahi_handle)
        

    def configure_search(self):
        search_phrase = "新型コロナ OR 新型肺炎 OR コロナ OR COVID  OR  (武漢 AND 肺炎)"
        search_box = self.driver.find_element("xpath", '//input')
        search_box.send_keys(search_phrase)
        search_btn = self.driver.find_element("xpath", '/html/body/div/div/div/section/section/main/section/section[2]/div[1]/div/form/div/div/div/div/button')
        self.click(search_btn)
        # TODO: finish configuration of search terms 
        print("configure search conditions")
    
    def crawl_search_results(self):
        dates = []
        titles = []
        articles = []
        go = True
        while go:
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            list_items = soup.find("ul", class_="md-topic-list")

            dates += [date.text for date in list_items.find_all("span", class_="md-topic-card-date")]
            with open("dates.txt", "w") as f:
                f.writelines("\n".join(dates))

            titles += [title.text for title in list_items.find_all("div", class_="md-topic-card-title")]
            with open("titles.txt", "w") as f:
                f.writelines("\n".join(titles))

            link_paths = '//ul/li/div/div[2]/a'
            links = self.driver.find_elements("xpath", link_paths)

            for link in links:
                try:
                    self.click(link)
                except Exception:
                    continue
                sleep(5)
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                article_text = soup.find("div", class_="md-article-card-detail__text")
                article_text = re.sub(r'<.*?>', '', str(article_text))
                articles.append(article_text)
                with open("articles.txt", "a") as f:
                    f.write(article_text + "\n\n")
            
            try:
                btn = self.driver.find_element(By.PARTIAL_LINK_TEXT, "次の")
                self.click(btn)
                sleep(10)
            except Exception:
                go = False
        
        asahi_df = pd.DataFrame({"date": dates, "title": titles, "text": articles})
        asahi_df.to_csv("./data/news/asahi.csv")
        

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


