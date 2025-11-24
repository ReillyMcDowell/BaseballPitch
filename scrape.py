import os
import time
import re
import requests
import pandas as pd

from typing import Optional
from bs4 import BeautifulSoup
from pybaseball import statcast_pitcher, playerid_lookup
from tqdm import tqdm  # Standard terminal progress bar

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urlparse, parse_qs

OUTPUT_FOLDER = "pitch_videos"

#Make sure output folder exists and if not makes the folder
def ensure_folder(folder: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)

#URL to scrape (all this is going to be very specific to baseballsavant's structure cause the Xpaths are tied to their HTML)
players_url = 'https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2025%7C2024%7C2023%7C2022%7C2021%7C2020%7C2019%7C2018%7C2017%7C&hfSit=&player_type=pitcher&hfOuts=&home_road=&pitcher_throws=&batter_stands=&hfSA=&hfEventOuts=&hfEventRuns=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&hfOpponent=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc#results'

#Getting the page to then put into BeautifulSoup to extract IDs
page=requests.get(players_url)
#Putting the page into BeautifulSoup to get the all the HTML IDs
soup = BeautifulSoup(page.content, 'html.parser')

#Making empty list to hold IDs and names
ids = []
ids_justnum = []
names = []
first_names = []
last_names = []
ids_df = pd.DataFrame(columns=['id', 'id_justnum', 'name', 'first_name', 'last_name'])
#Using BeautifulSoup to find all the 'td' elements and extract their IDs and names
for td in soup.find_all('td'):
    if td.has_attr('id'):
        ids.append(td.get('id'))
        names.append(td.get_text().replace('\n                                    ', '').replace(' \n', '').replace(' LHP', '').replace(' RHP', ''))
ids_df['id'] = ids
ids_df['name'] = names
ids_justnum = [id.replace('id_', '') for id in ids]
ids_df['id_justnum'] = ids_justnum
first_names = [name.split(', ')[1] for name in names]
ids_df['first_name'] = first_names
last_names = [name.split(', ')[0] for name in names]
ids_df['last_name'] = last_names

#ids_not_found = []
#ids_not_match = []
#for i in range(len(names)):
#    if len(playerid_lookup(last = ids_df['last_name'][i], first = ids_df['first_name'][i]).key_mlbam) == 0:
#        ids_not_found.append(i)
#    else:
#        if playerid_lookup(last = ids_df['last_name'][i], first = ids_df['first_name'][i]).key_mlbam[0] != int(ids_df['id_justnum'][i]):
#            ids_not_match.append(i)
#print(ids_not_found)
#print(ids_not_match)
#set(ids_not_found) & set(ids_not_match)

#print(f"Collected {len(ids)} IDs.") # Debugging how many IDs found
#print(ids) # Debugging to see the list of IDs (their structure is 'id_XXXXXX' where XXXXXX is a number)
if len(ids) == 0:
    print("No IDs found on the page. Exiting.")
    exit(1)
if len(names) == 0:
    print("No names found on the page. Exiting.")
    exit(1)

#Setting up Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

search_url = "https://baseballsavant.mlb.com/statcast_search"

print(f"Navigating to: {search_url}")
#Selenium to get the page
driver.get(search_url)

# Edit fields, we want one player to reduce load times and we want only the years with statcast
# 1. Season Button (ensure it's clickable/visible)
season_btn = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, "//div[@id='boxSea']"))
)
# Scroll to it just in case
driver.execute_script("arguments[0].scrollIntoView(true);", season_btn)
# Click
season_btn.click()
#driver.save_screenshot('SeasonClicked.png') # Debugging to make sure it's the right page after click
print("Clicked Season. Waiting for results...")
#time.sleep(1)  # brief pause to let any dynamic suggestions load
# 2. Statcast Button (ensure it's clickable/visible)
statcast_btn = driver.find_element(By.XPATH, "//span[@id='year_statcast']")
# Click
statcast_btn.click()
#driver.save_screenshot('StatcastClicked.png') # Debugging to make sure it's the right page after click
print("Clicked Statcast. Waiting for results...")
# 3. Player Name Button and type some text
player_name_btn = driver.find_element(By.XPATH, "//textarea[@aria-describedby='select2-pitchers_lookup-container']")
# Scroll to it just in case
driver.execute_script("arguments[0].scrollIntoView(true);", player_name_btn)
# Enter Player Name
player_name_btn.send_keys(ids_df['name'][0])
#driver.save_screenshot('PlayerEntered.png') # Debugging to make sure it's the right page after entering
print(f"Entered Player Name: {ids_df['name'][0]}")
player_autofill_btn = driver.find_element(By.XPATH, "//ul[@id='select2-pitchers_lookup-results']/li[1]")
# Click the auto-fill suggestion
player_autofill_btn.click()
print("Clicked First Autofill Suggestion. Waiting for results...")
#driver.save_screenshot('PlayerClicked.png') # Debugging to make sure it's the right page after click
time.sleep(1)  # brief pause to let any dynamic suggestions load
# 4. Find the Search button (ensure it's clickable/visible)
# Used XPath tester to find the correct XPath
search_btn = WebDriverWait(driver, 20).until(
    EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))
)
# Scroll to it just in case
driver.execute_script("arguments[0].scrollIntoView(true);", search_btn)
# Click
search_btn.click()
#driver.save_screenshot('SearchClicked.png') # Debugging to make sure it's the right page after click
print("Clicked Search. Waiting for results...")
# 5. Find the first player in our table (ensure it's clickable/visible)
# using the first ID from our previously collected list
# would want to change the index [0] to [n] where n is the row number minus one for making a loop for all players
first_link = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, f"//td[@id='{ids_df['id'][0]}']"))
)
# Scroll to it just in case
driver.execute_script("arguments[0].scrollIntoView(true);", first_link)
# Click
first_link.click()
print("Clicked First Link. Waiting for results...")
# 6. Verify navigation
# Wait for url to change or page to load
time.sleep(10) # brief pause to let nav happen
#driver.save_screenshot('FirstLinkClicked.png') # Debugging to make sure it's the right page after click

video_href: str | None = None  # ensure variable is bound
try:
    # 7. Find the video (not clicking it here, just taking the href from the XPath for the video button)
    # XPath for the video button in the first row of the table tr[1] and td[15] is the video column
    # For future where one would want to change the row, just change the tr[1] to tr[n] where n is the row number
    video_link = driver.find_element(By.XPATH, f"//table[@id='{ids_df['id'][0].replace('id', 'ajaxTable')}']/tbody/tr[1]/td[15]/a")
    href = video_link.get_attribute('href')
    print(f"Video Link href: {video_href}")
    if isinstance(href, str):
        video_href = href
    print(f"Video Link href: {video_href}")
    print(f"New Page URL: {driver.current_url}")
except Exception as e:
    print(f"Error locating video link: {e}")
finally:
    driver.quit()

# Want to take the video link href and use it to download the video file directly to pitch_videos folder

# Guard before using requests.get (part of the whole bounding thing for our href)
if video_href is None:
    print("No video href extracted; skipping video fetch.")
else:
    #Getting the page to then put into BeautifulSoup to extract mp4s
    page=requests.get(video_href)
    #Putting the page into BeautifulSoup to get the all the HTML for the mp4 URL
    soup = BeautifulSoup(page.content, 'html.parser')

    mp4_url = None
    #Using BeautifulSoup to find all the 'source' elements and extract their mp4 URLs
    for source in soup.find_all('source'):
        if source.has_attr('src'):
            raw_src = source.get('src')
            if isinstance(raw_src, list):
                mp4_url = raw_src[0] if raw_src else None
            elif isinstance(raw_src, str):
                mp4_url = raw_src
            break  # Use first valid source
        
    if mp4_url is None:
        print("No mp4 URL found in source tag.") 
    # Downloading the mp4 file into the OUTPUT_FOLDER
    else:
        # Defined above but makes sure output folder exists
        ensure_folder(OUTPUT_FOLDER)
        placeholder_mp4_name = 'pitch_video'
        filename = f"{placeholder_mp4_name}.mp4"
        path = os.path.join(OUTPUT_FOLDER, filename)
        try:
            # Stream = True to download large files by splitting it into chunks to download (instead of all at once)
            r = requests.get(mp4_url, stream=True, timeout=30)
            # HTTP response status code check (200 is success so anything else is bad)
            if r.status_code != 200:
                print(f"[DOWNLOAD] Video request failed {r.status_code} for {mp4_url}")
            else:
                # Writing the file in binary mode cause it's a video (hence the 'wb') and f as file object
                with open(path, 'wb') as f:
                    # Writing in chunks to handle large files, 8192 bytes at a time (8 KB)
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"[DOWNLOAD] Saved video to {path}")
        except Exception as e:
            print(f"[DOWNLOAD] Exception downloading video {mp4_url}: {e}")
