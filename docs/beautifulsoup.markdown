---
layout: page
title: Beautiful Soup
category: webscraping
permalink: /BeautifulSoup/
order: 1
---

## Why not just use Beautiful Soup for our webscraping efforts?

Beautiful Soup is a Python package that can help parse html. It can even find the links for videos in website pages!

{% highlight python %}
import requests
from bs4 import BeautifulSoup

player_url = "https://baseballsavant.mlb.com/statcast_search"
# Getting the page to then put into BeautifulSoup to extract IDs
page = requests.get(players_url)
# Putting the page into BeautifulSoup to get the all the HTML IDs
soup = BeautifulSoup(page.content, 'html.parser')

{% endhighlight %}

This sounds great, so why are we not using it?

Beautiful Soup is limited in the html it can parse. It **cannot parse dynamic content** - so any clickable menus or tables that we need to search for a single player's pitch video won't be in our html soup.

We need something that can go in a fill out menus, search fields, and click the tab to open the video link...

[Selenium can solve this!]{https://reillymcdowell.github.io/BaseballPitch/Selenium/}
