---
layout: page
title: Selenium
category: webscraping
permalink: /Selenium/
order: 2
---

## Selenium Webdriver

We need something that can go in a fill out menus, search fields, and click the tab to open the video link on Baseball Savant.

Selenium opens a webdriver to browse a given site. It has commands for basic actions like finding elements, clicking them, or sending keys. 

*Basic Selenium Example:*
{% highlight python %}
from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize browser
driver = webdriver.Chrome()
driver.get("https://www.google.com")

# Interact with elements
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("Selenium Example")
search_box.submit()

# Close browser
driver.quit()

{% endhighlight %}

Awesome. We can actually set up a lot of these steps that we want to accomplish as individual functions that we can loop through later.

Specifically on Baseball Savant there are a lot of fields for filtering what you want to search. There are not always pitch videos for the years before statcast, despite having other metrics recorded, so a first good step can be writing a filtering function for our webdriver.

*Basic Selenium Example:*
{% highlight python %}
def season_filtering():
    """
    Function to click the Season button and select Statcast
    """
    # Edit fields, we want one player to reduce load times
    # and we want only the years with statcast
    
    # 1. Season Button (ensure it's clickable/visible)
    season_btn = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//div[@id='boxSea']"))
    )
    # Scroll to it just in case
    driver.execute_script("arguments[0].scrollIntoView(true);", season_btn)
    # Click
    season_btn.click()
    # Debugging to make sure it's the right page after click
    # driver.save_screenshot('SeasonClicked.png')
    print("Clicked Season. Waiting for results...")
    # time.sleep(1)  # brief pause to let any dynamic suggestions load
    
    # 2. Statcast Button (ensure it's clickable/visible)
    statcast_btn = driver.find_element(By.XPATH, "//span[@id='year_statcast']")
    # Click
    statcast_btn.click()
    # Debugging to make sure it's the right page after click
    # driver.save_screenshot('StatcastClicked.png')
    print("Clicked Statcast. Waiting for results...")
    pass

{% endhighlight %}

Our other filtering functions follow a similar format. Something to note here is that Selenium is blind in a way - **it sees via HTML and XPaths**. 

![XPath Example](XPath_example.jpg)
