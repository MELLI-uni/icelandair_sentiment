import pandas as pd
import requests
from bs4 import BeautifulSoup

import re
import time
from datetime import datetime

def destination():
    """
    destination function scrapes information on the routes icelandair travels through and saves it the flight-city.txt file

    : param: None
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

def emoji():
    URL = "https://emojipedia.org/"

    print("emoji scraping")

emoji()