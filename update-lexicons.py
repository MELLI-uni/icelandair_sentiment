import numpy
import pandas as pd
import requests
from bs4 import BeautifulSoup

import shutil
import os
import re
import time
from datetime import datetime

import UpdateLanguage

def update_emoticons():
    """
    Convert western emoticon into a text file

    Emoticon list from:
        https://en.wikipedia.org/wiki/List_of_emoticons
        Section: Western, Sideways Latin-only emoticons
        
    Requirements:
        File_name should be setted to "western-emoticons.xlsx"
        All data must be placed on "Sheet1"
        Cells of excel MUST be formatted to 'text' when copying and pasting
    """

    file_name = "western-emoticons.xlsx"
    sheet = "Sheet1"
    df = pd.read_excel(io = file_name, sheet_name = sheet)

    # Eliminate all reference mark in format [#] and more than one spaces
    # Copies preceding row for meaning if empty
    df_new = df["Meaning"].str.replace(r'(\[[0-9]+\])+', '', regex=True).str.replace(r'\s+', ' ', regex=True).ffill()
    del df["Meaning"]

    # Converts every row of emoticons to one list and remove all 'nan' from the list
    df["Emoticon"] = df.values.tolist()
    df["Emoticon"] = df["Emoticon"].apply(lambda x: [i for i in x if (pd.isnull(i)) == False])

    # Append "Meaning" column to the end and drop all columns except "Emoticon" and "Meaning"
    df["Meaning"] = df_new
    df.drop(df.columns.difference(['Emoticon', 'Meaning']), axis=1, inplace=True)

    # Explode the "Emoticon" column
    df = df.explode('Emoticon')

    df.to_csv('./lexicons/emoticon.txt', header=None, index=None, sep='\t', mode='w')

def update_destination():
    """
    Scrape information on icelandAir's routes into textfile

    Content from:
        https://www.icelandair.com/support/airports/

    Requirements:
        Internet connection is needed to scrape data
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

    flight_city_file = "./lexicons/destination.txt"
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

def update_emojis():
    """
    Scrapes newest emojis and append to pre-existing emoji file

    Emoji from:
        https://emojipedia.org
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

#####################################################################################
def EtcLexiconUpdate(destination=True, emoticons=False, emojis=True):
    if destination == True:
        update_destination()

    if emoticons == True:
        update_emoticons()

    if emojis == True:
        update_emoticons()

def LanguageLexiconUpdate(excel_file, excel_sheet):
    UpdateLanguage.update(excel_file, excel_sheet)