# nikkei_crawler.py: Crawls nikkei shinbun database for articles with the following search terms:
# 新型　OR 感染
# in their headlines. 
# To use, open https://waseda-jp.libguides.com/az.php?q=%E6%97%A5%E6%9C%AC%E7%B5%8C%E6%B8%88%E6%96%B0%E8%81%9E&p=1, 
# login using WasedaID, and configure search parameters. I searched for Articles in Nikkei shinbun Morning print
# with topic keywords  [一般用語:新型] AND [一般用語:感染]　between 2020-01-01 and 2021-12-31

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
from selenium.webdriver.support.ui import Select

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
        input("Login to your Waseda Account and press <ENTER> to continue: ")

        # switch to nikkei tab
        original_handle = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != original_handle:
                self.driver.switch_to.window(handle) # nikkei tab


    def configure_search(self):
        # go to article search. Skip if you want to crawl non articles 
        article_search_btn = self.find_element(By.XPATH, '//*[@id="nk-mainmenu"]/div/div[1]/div/ul/li[2]/p')
        self.click(article_search_btn)
        sleep(3)
        input("Configure search parameters and press <ENTER> to continue: ")
        # show 400 results per page 
        show_400_btn = Select(self.find_element(By.XPATH, '//div[@class="nk-pn-info-search-result nk-np-info-search-result-up"]/div[@class="nk-pn-info-sc"]/select[@name="cnt"]'))
        show_400_btn.select_by_visible_text("400")
        # show articles 
        show_btn = self.find_element(By.XPATH, '//button[@name="listUp"]')
        self.click(show_btn)
        # confirm show articles in pop up 
        confirm_btn = self.find_element(By.XPATH, '//button[@class="nk-popup-ok"]')
        self.click(confirm_btn)
        sleep(20)
        
    
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
        
        # iterate while go 
        go = True
        while go:
            # select all articles shown in page
            select_all_btn = self.driver.find_element("xpath", '//div[@class="nk-navigator-select-all"]')
            self.click(select_all_btn)
            sleep(5)

            # open all articles
            open_all_btn = self.driver.find_element("xpath", '//button[@class="nk-list-navigator-honbun"]')
            self.click(open_all_btn)
            sleep(1)

            # press confirm popup
            confirm_btn = self.driver.find_element("xpath", '//button[@class="nk-popup-ok"]')
            self.click(confirm_btn)
            sleep(100)

            # get html and build soup 
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # parse soup to find all article titles
            title_blocks = soup.find_all("h2", class_="nk-gv-bodytitle")
            titles_temp = [title_block.text for title_block in title_blocks]
            with open("titles.txt", "a") as f:
                f.writelines("\n".join(titles_temp) + "\n")

            # parse soup to find all article dates
            subtitle_blocks = soup.find_all("div", class_="nk-gv-attribute")
            dates_temp = [subtitle_block.text[:10] for subtitle_block in subtitle_blocks]
            with open("dates.txt", "a") as f:
                f.writelines("\n".join(dates_temp) + "\n")

            # parse soup to find all articles
            # xpath //td[@class="nk-gv-body-view nk-gv-artbody nk-gv-artbody-honbun"]//td
            article_blocks = soup.find_all("td", class_="nk-gv-body-view nk-gv-artbody nk-gv-artbody-honbun")
            articles_temp = [article_block.find("td").text.replace(",", "") for article_block in article_blocks]
            with open("articles.txt", "a") as f:
                f.writelines("\n".join(articles_temp) + "\n")

            # add temporary titles, articles, dates
            dates += dates_temp
            titles += titles_temp
            articles += articles_temp

            self.driver.back()
            sleep(5)
            
            try:
                btn = self.driver.find_element(By.XPATH, '//li[@class="nk-navigator-next"]')
                self.click(btn)
                sleep(30)
            except Exception:
                go = False
        
        nikkei_df = pd.DataFrame({"date": dates, "title": titles, "text": articles})
        nikkei_df["date"] = pd.to_datetime(nikkei_df["date"], format="%Y/%m/%d")
        nikkei_df.to_csv("./data/news/nikkei.csv", index=False)
        

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
    n = NikkeiDriver()
    n.main()

if __name__=="__main__":
    main()
