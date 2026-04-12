---
layout: page
title: Baseball Pitch
---

## Project Description

Aim to perform real-time Baseball Pitch Classification (what type of pitch is being thrown). 

## Webscrapping

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
    pass  # Placeholder for potential future use
{% endhighlight %}

# What type of thing is this?
