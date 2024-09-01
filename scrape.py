import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests as rq
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager
import datetime  # Import datetime for timestamp

def scrape_airline_reviews():
    # Initialize lists to store scraped data
    verified, rating, feedback, Aircraft, Class, Travel_type, Route, Date = ([] for _ in range(8))
    comfort, staff, food, entertainment, wifi, ground_service, vfm, recommend, airline_name = ([] for _ in range(9))

    # Function to fetch values based on tag and attributes
    def fetcher(soup, tag, key, value):
        element = soup.find(tag, {key: value})
        return element.text if element else np.nan

    # Function to extract star ratings and other review details
    def value_fetcher(soup):
        rating.append(np.nan)
        Aircraft.append(np.nan)
        Class.append(np.nan)
        Travel_type.append(np.nan)
        Route.append(np.nan)
        Date.append(np.nan)
        comfort.append(np.nan)
        staff.append(np.nan)
        food.append(np.nan)
        entertainment.append(np.nan)
        wifi.append(np.nan)
        ground_service.append(np.nan)
        vfm.append(np.nan)
        recommend.append(np.nan)

        for row in soup.find_all('tr'):
            header = row.find('td', class_='review-rating-header').text
            value = row.find('td', class_='review-value')
            if header == 'Aircraft':
                Aircraft[-1] = value.text
            elif header == 'Type Of Traveller':
                Travel_type[-1] = value.text
            elif header == 'Seat Type':
                Class[-1] = value.text
            elif header == 'Route':
                Route[-1] = value.text
            elif header == 'Date Flown':
                Date[-1] = value.text
            elif header == 'Seat Comfort':
                comfort[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Cabin Staff Service':
                staff[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Food & Beverages':
                food[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Inflight Entertainment':
                entertainment[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Ground Service':
                ground_service[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Value For Money':
                vfm[-1] = len(row.find_all('span', class_='star fill'))
            elif header == 'Recommended':
                recommend[-1] = value.text
            elif header == 'Wifi & Connectivity':
                wifi[-1] = len(row.find_all('span', class_='star fill'))

    # Function to get airline names using Selenium
    def get_airline_names(url):
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        driver.get(url)
        time.sleep(3)
        ul_elements = driver.find_elements(By.XPATH, '/html/body/div[1]/div[1]/div/div/section[2]/div[1]/section/ul')
        list_ = []
        for ul in ul_elements:
            try:
                body_divs = ul.find_elements(By.CLASS_NAME, 'body')
                for body_div in body_divs:
                    airline_name = body_div.find_element(By.TAG_NAME, 'a').text
                    list_.append(airline_name)
            except Exception as e:
                print(f"Error: {e}")
                continue
        driver.quit()
        return list_

    # Main scraping process
    airline_list = get_airline_names('https://www.airlinequality.com/airline-reviews')
    len_list = len(airline_list)
    index_end = int(input(f'Input the number of airlines you want to gather data from, max is {len_list}: '))
    pages = int(input("Input the number of pages you want to read for each of the airlines: "))
    airline_list = airline_list[0:index_end]
    for airline in airline_list:
        print(f"Scraping from: {airline}")
        for i in range(1, pages+1):
            print(f"Page {i}")
            airline = airline.lower().replace(" ", "-")
            url = f'https://www.airlinequality.com/airline-reviews/{airline}/page/{i}/'
            page = rq.get(url).text
            soup = BeautifulSoup(page, 'lxml')
            articles = soup.find_all('article', {'itemprop': 'review'})
            for article in articles:
                airline_name.append(airline)
                rating.append(fetcher(article, 'span', 'itemprop', 'ratingValue'))
                div = article.find('div', class_='tc_mobile')
                content = div.find('div', class_='text_content').text.split('|')
                if len(content) == 2:
                    verification, review = content
                else:
                    verification = np.nan
                    review = content[0]
                verified.append(verification)
                feedback.append(review)
                table = div.find('table', class_='review-ratings')
                value_fetcher(table)

    # Creating a DataFrame and saving the data
    data = pd.DataFrame({
        'status': verified,
        'aircraft': Aircraft,
        'travel_type': Travel_type,
        'travel_class': Class,
        'route': Route,
        'date': Date,
        'seating_comfort': comfort,
        'staff_service': staff,
        'food_quality': food,
        'entertainment': entertainment,
        'wifi': wifi,
        'ground_service': ground_service,
        'value_for_money': vfm,
        'recommended': recommend,
        'overall_rating': rating[::2],
        'review': feedback,
        'airline_name': airline_name
    })

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data_filtered = data[data['date'].dt.year >= 2024]

    # Adding NPS_score column based on overall_rating
    def determine_nps_score(rating):
        if pd.isna(rating):
            return np.nan
        rating = float(rating)
        if rating >= 8:
            return 'Promoter'
        elif 6 <= rating <= 7:
            return 'Neutral'
        elif rating <= 5:
            return 'Detractor'
        else:
            return np.nan

    data_filtered['NPS_score'] = data_filtered['overall_rating'].apply(determine_nps_score)

    # Adding the current date to the filename
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f'data/raw_scraped_{today_date}_filtered.csv'
    data_filtered.to_csv(filename, index=False)
    
    return data_filtered
