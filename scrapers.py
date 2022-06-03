import pandas as pd
import requests
from bs4 import BeautifulSoup

import shutil
import os
import re
import time
from datetime import datetime

def destination():
    """
    destination function scrapes information on the routes icelandair travels through and saves it the flight-city.txt file

    : return: None
    """

    # download content
    URL = "https://www.icelandair.com/support/airports/"
    data = requests.get(URL)

    # create soup object
    soup = BeautifulSoup(data.text, 'lxml')
    table = soup.find('table', class_='airports_table__2dArS')

    # initialize airport code list and airport list
    airport_code = []
    airport = []

    # collecting data
    for row in table.tbody.find_all('tr'):
        
        # find the column with data-label 'Airport Code: ' and append to airport_code list
        code_column = row.find_all('td', attrs={'data-label': 'Airport Code:'})
        for i in code_column:
            airport_code.append(str(i.text))

        # find the column with data-label 'Airport: '
        for airport_column in row.find_all('td', attrs={'data-label': 'Airport:'}):
            temp = "temp"

            # find the name of airport and append variants of airport name to airport list
            name_column = airport_column.find_all('a')
            for i in name_column:
                airport.append(str(i.text))
                if "Airport" in i.text:
                    temp = re.split("Airport", i.text)[0].strip()
                    airport.append(temp)
                if "International Airport" in i.text:
                    temp = re.split("International Airport", i.text)[0].strip()
                    airport.append(temp)

            # find the name of the city and append the name to airport list if it is different with a possible variant of airport name
            city_column = airport_column.find_all('span')
            for i in city_column:
                if(str(i.text) != temp):
                    airport.append(str(i.text))

    flight_city_file = "./lexicons/flight-city.txt"
    with open(flight_city_file, 'w', encoding='utf-8') as f:
        # Write the last updated date/time in datetime format
        timestamp = time.time()
        dt = datetime.fromtimestamp(timestamp)
        f.write("Last Updated: " + str(dt))

        for codes in airport_code:
            f.write(codes + "\n")

        for ports in airport:
            f.write(ports + "\n")

    f.close()

def emoji_update():
    """
    emoji_update function scrapes and appends new emojis to end of file if there is a new version
 
    : return: None
    """

    # open emoji file and get the last updated version
    emoji_file = "./lexicons/emoji.txt"
    f = open(emoji_file, "r", encoding="utf-8")
    emoji_file_version = f.readline().rstrip()

    # download content
    URL = "https://emojipedia.org"
    data = requests.get(URL)

    # create soup object
    soup = BeautifulSoup(data.text, 'lxml')
    emoji_versions = soup.find('footer', class_='page-footer').find('div', class_='unicode-version').find_all('ul')[0]
    current_version = emoji_versions.find_all('li')[1]

    # grab the most recent url_extension and version_name from the footer
    url_extension = current_version.find('a').get("href")
    version_name = current_version.text
    
    # End the program if the emoji version listed on the emoji lexicon is the same as the most recent emoji version on emojipedia
    if emoji_file_version == version_name:
        f.close()
        print("No Updates Required")

        exit()
    else: 
        f_temp = open("temp.txt", "w", encoding="utf-8")
        
        # copy all the emojis in the previous file to a new file
        f_temp.write(version_name + "\n")
        shutil.copyfileobj(f, f_temp)
        f.close()
        
        URL = URL + url_extension
        data = requests.get(URL)

        soup = BeautifulSoup(data.text, 'lxml')
        finding = soup.find('div', class_='content').find_all('ul')[0]

        # append all new emojis available in the new version
        for new_emoji in finding.find_all('li'):
            line = new_emoji.find('a').text.split(" ", 1)
            icon = line[0]
            description = line[1]
            f_temp.write(icon + "\t" + description + "\n")
        f_temp.close()

        # replace the original file
        shutil.move("temp.txt", emoji_file)

def emoji_all():
    """
    emoji_all function scrapes every emoji available on emojipedia

    : return: None
    """
    # open emoji file and get the last updated version
    emoji_file = "./lexicons/emoji.txt"
    f = open(emoji_file, "w", encoding="utf-8")
    
    # download content
    URL = "https://emojipedia.org"
    data = requests.get(URL)

    # create soup object
    soup = BeautifulSoup(data.text, 'lxml')
    emoji_versions = soup.find('footer', class_='page-footer').find('div', class_='unicode-version').find_all('ul')[0]

    # write the most recent emoji version available on emojipedia
    f.write(emoji_versions.find_all('li')[1].text + "\n")

    # scrape all the emojis in the format of <emoji>\t<description>
    for i in emoji_versions.find_all('li')[1:]:
        url_extension = i.find('a').get("href")

        new_URL = URL + url_extension
        data = requests.get(new_URL)

        soup = BeautifulSoup(data.text, 'lxml')
        finding = soup.find('div', class_='content').find('h2').findNext('ul')
        for new_emoji in finding.find_all('li'):
            line = new_emoji.find('a').text.split(" ", 1)
            icon = line[0]
            description = line[1]
            f.write(icon + "\t" + description + "\n")

    f.close()