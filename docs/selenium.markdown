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

*Example:*
{% highlight python %}
from selenium import webdriver
from selenium.webdriver.common.by import By

# Initialize browser
driver = webdriver.Chrome()
driver.get("https://www.google.com")

# Interact with elements
search_box = driver.find_element(By.NAME, "q")
search_box.send_keys("Selenium Automation")
search_box.submit()

# Close browser
driver.quit()

{% endhighlight %}


