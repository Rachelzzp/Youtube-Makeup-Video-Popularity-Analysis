import time
#import pandas as pd
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import csv
import pandas as pd
import requests
# chrome_options = Options()
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# driver = webdriver.Chrome('/usr/bin/chromedriver',chrome_options=chrome_options)
#driver.get('https://www.youtube.com/results?search_query=makeup&sp=CAMSBAgFEAE%253D')
#driver = webdriver.Chrome("/usr/local/bin/chromedriver")
#driver.get("https://www.youtube.com/results?search_query=makeup&sp=CAASAhAC") #makeup channel
#driver.get("https://www.youtube.com/results?search_query=beauty&sp=CAASAhAC") #beauty channel           
            
c_link = "https://www.youtube.com/channel/UCHQO9xPSROEHYHUk_T82l6w/videos"
#cName = "channel/UCucot-Zp428OwkyRm2I7v2Q"
print(c_link)

# chrome_options = Options()
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--disable-dev-shm-usage')
# driver_second= webdriver.Chrome('/usr/bin/chromedriver',chrome_options=chrome_options)
driver_second = webdriver.Chrome("/usr/local/bin/chromedriver")
driver_second.get(c_link)

csvFile = open("channel_each_12.csv", "w")
csvWriter = csv.writer(csvFile)
csvWriter.writerow(["ID","link","title","viewcount","like","dislike","publish_date","description"])

old_position_2 = 0
new_position_2 = None
len2 = 0
#while new_position != old_position:
source_second = driver_second.page_source    
soup_v = BeautifulSoup(source_second,"lxml")
while new_position_2!= old_position_2:
    
        #time.sleep(1)
    old_position_2 = driver_second.execute_script(
        ("return (window.pageYOffset !== undefined) ?"
        " window.pageYOffset : (document.documentElement ||"
        " document.body.parentNode || document.body);"))
    # Sleep and Scroll
    time.sleep(3)

    driver_second.execute_script((
            "var scrollingElement = (document.scrollingElement ||"
            " document.body);scrollingElement.scrollTop ="
            " scrollingElement.scrollHeight;"))
    # Get new position
    new_position_2 = driver_second.execute_script(
            ("return (window.pageYOffset !== undefined) ?"
            " window.pageYOffset : (document.documentElement ||"
            " document.body.parentNode || document.body);"))
    len2 +=1
    print("-----------------    IINSIDE VIDEO OCUNT LEN :",len2, "-------------------")
source_second = driver_second.page_source    
soup_v = BeautifulSoup(source_second,"lxml")
for each_v in soup_v.findAll("div",{"id":"details"}):
    # with open("soup_cchannel.txt", "w") as file:
    #     file.write(str(soup_v))
    title = each_v.find("a", {"id":"video-title"}).text
    title.replace(" ", "")
    print("TITLE: " , title)
    v_url = each_v.find("a")["href"]
    v_ID = v_url[9:]
    print(v_ID)
    v_url = "https://www.youtube.com"+v_url
    v_url.replace(" ", "")
    print(v_url)

    # chrome_options = Options()
    # chrome_options.add_argument('--headless')
    # chrome_options.add_argument('--no-sandbox')
    # chrome_options.add_argument('--disable-dev-shm-usage')
    # driver_third = webdriver.Chrome('/usr/bin/chromedriver',chrome_options=chrome_options)
    third_data = requests.get(v_url)
    #print(source_data)
    soup_third = BeautifulSoup(third_data.text,"lxml")
        #rep_v_second = requests.get(v_url)
        #soup_v_second = BeautifulSoup(rep_v_second.text, "lxml")
    try:
        # with open("soup_cchannel.txt", "w") as file:
        #     file.write(str(soup_third))
        #     exit()
        viewcount = soup_third.find("div", class_="watch-view-count").get_text()
        print("WATCH COUNT_ ", viewcount)
        like = soup_third.find("button",class_="yt-uix-button yt-uix-button-size-default yt-uix-button-opacity yt-uix-button-has-icon no-icon-markup like-button-renderer-like-button like-button-renderer-like-button-clicked yt-uix-button-toggled hid yt-uix-tooltip").find("span",class_="yt-uix-button-content").get_text()
        print("LIKE_",like)
        dislike = soup_third.find("button",class_="yt-uix-button yt-uix-button-size-default yt-uix-button-opacity yt-uix-button-has-icon no-icon-markup like-button-renderer-dislike-button like-button-renderer-dislike-button-clicked yt-uix-button-toggled hid yt-uix-tooltip").find("span",class_="yt-uix-button-content").get_text()
        print("DISLIKE_", dislike)   
        publish_date = soup_third.find("div",id="watch-uploader-info").find("strong",class_="watch-time-text").get_text()
        print("PUBLISH DATE_ ",publish_date)        
        description = soup_third.find("p", id="eow-description").get_text()
        #print("[DESCRIPTION_] ",description)
        csvWriter.writerow([v_ID,v_url,title,viewcount,like,dislike,publish_date,description])
    except AttributeError:
        print("this video is the error")

driver_second.close()