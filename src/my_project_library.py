import pandas as pd
import numpy as np
from edgar import Company
from edgar import TXTML
import unicodedata
import requests
from bs4 import BeautifulSoup
import lxml.html
import lxml.html.soupparser
import requests, re
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import yfinance as yf
import datetime as dt
import datetime
import calendar

def pull_10K(name, company_id):
    '''
    we use this function to perform the get filings.
    we need to run this function and iterarte over our
    list of tickers. Each ticker will get parsed and
    collected into a dataframe.
    '''
    company = Company(name, company_id)
    tree = company.get_all_filings(filing_type = "10-K")

    docs = Company.get_documents(tree, no_of_documents=6)
    # print("checkpoint: retrieving documents...")
    text_l=[]
    for i in range(len(docs)):
        try:
            text=TXTML.parse_full_10K(docs[i])
            text_l.append(text)
        except IndexError:
            pass
    return text_l


def pull_risk_section(text):

    '''
    this function will parse each 10k and
    use regex to find the Risk Factors section
    '''
    text = re.sub('\n', ' ', text)
    text = re.sub('\xa0', ' ', text)
    matches = list(re.finditer(re.compile('Item [0-9][A-Z]\.', re.IGNORECASE), text))
    #using this bock of code to isolate any null values
    if matches==[]: #avoid returning empty sets for 10k ammendments that may exclude 1A sections
        return text
    try:
        start = max([i for i in range(len(matches)) if matches[i][0].casefold() == ('Item 1A.').casefold()])
    except:
        return text #avoid returning empty sets for 10k ammendments that may exclude 1A sections
    #print(text)
    end = start+1
    start = matches[start].span()[1]
    if end > len(matches)-1:
        end = start
    else:
        end = matches[end].span()[0]
    text = text[start:end]
    #print(start, end)

    return text


def file_date(com, cik, no_docs):
    """
    This function is to pull only the filing date
    Serves as the date of measurement for analyzing returns.
    """
    company = Company(com, cik, no_docs)
    tree = company.get_all_filings(filing_type = "10-K")
    docs = Company.get_documents(tree, no_of_documents=no_docs, as_documents=True)
    dates = []
    for x in range(no_docs):
        doc = docs[x]
        dates.append(doc.content['Filing Date'])

    return dates

def scrape_wiki(url):
    """
    Simple code to parse wikipedia table
    will adjust here if any changes occur to wiki
    """
    scrape_url = requests.get(url).text
    soup = BeautifulSoup(scrape_url, 'xml')

    return(soup)

def word_filter():
    """
    Financial sentiment classification from  LM Dictionary(Tim Loughran and Bill McDonald)
    Focused on the negative sentiment and will apply this matrix processed from excel file below.
    """
    word_list = []
    for sentiment_class in ["Negative", "Uncertainty", "Litigious","StrongModal"]:
        #for jupyter notebook, use the local Word List.xlsx location
        sentiment_list = pd.read_excel("/home/gregjroberts1/Projects/Data/LM_Word_List.xlsx",sheet_name=sentiment_class,header=None)
        sentiment_list.columns = ["Word"]
        sentiment_list["Word"] = sentiment_list["Word"].str.lower()
        sentiment_list[sentiment_class] = 1
        sentiment_list = sentiment_list.set_index("Word")[sentiment_class]
        word_list.append(sentiment_list)

    word_list = pd.concat(word_list, axis=1, sort=True).fillna(0)
    negative_words = word_list.index
    return negative_words


# Define the weekday mnemonics to match the date.weekday function
(MON, TUE, WED, THU, FRI, SAT, SUN) = range(7)
# Define default weekends, but allow this to be overridden at the function level
default_weekends=(SAT,SUN)
def workday(start, days, holidays=[], weekends=default_weekends):
    """
    Repurposed to suit my needs for date processing business days from a periodic format
    Then casted back into a periodic form to process with Yahoo Finance returns
    -- started from the code of Casey Webster at
    http://groups.google.com/group/comp.lang.python/browse_thread/thread/ddd39a02644540b7
    """

    sd= start.strftime("%Y-%m-%d")
    start_date = datetime.datetime.strptime(sd,'%Y-%m-%d')

    if days == 0:
        return start_date;
    if days>0 and start_date.weekday() in weekends: #
      while start_date.weekday() in weekends:
          start_date -= timedelta(days=1)
    elif days < 0:
      while start_date.weekday() in weekends:
          start_date += timedelta(days=1)
    full_weeks, extra_days = divmod(days,7 - len(weekends))
    new_date = start_date + timedelta(weeks=full_weeks)
    for i in range(extra_days):
        new_date += timedelta(days=1)
        while new_date.weekday() in weekends:
            new_date += timedelta(days =1)
    # to account for days=0 case
    while new_date.weekday() in weekends:
        new_date += timedelta(days=1)
    # avoid this if no holidays
    if holidays:
        delta = timedelta(days=1 * cmp(days,0))
        # skip holidays that fall on weekends
        holidays =  [x for x in holidays if x.weekday() not in weekends ]
        holidays =  [x for x in holidays if x != start_date ]
        for d in sorted(holidays, reverse = (days < 0)):
            # if d in between start and current push it out one working day
            if _in_between(start_date, new_date, d):
                new_date += delta
                while new_date.weekday() in weekends:
                    new_date += delta
    new_date=new_date.strftime("%Y-%m-%d")
    new_date = pd.to_datetime(new_date, format='%Y-%m-%d').to_period(freq ='d')
    return new_date

def get_px_chgs(today,ticker_select, start, end):
    """
    created to retrieve prices from Yahoo finance function.
    needs the start (filing date) and end date(assigned in the main program at 200_days from filing)
    despite only looking at 90 days, having the flexibility to extend to 6mos if data warrants
    calculating an average price given the high,low, and close.  ideally want the VWAP - calc in future update
    """
    print(ticker_select) #maintain status of ticker processing in case of error in data
    ticker_prices = yf.download(ticker_select, start=start, end=end, auto_adjust=False) #run for each filing on each ticker -  4 filings per ticker
    # ticker_prices['AvgPx'] = ticker_prices.iloc[:,1:4].sum(axis=1) / 3  #average of high, low, close - calc vwap?

    ticker_prices.reset_index(inplace=True)
    adj_close, date = [], []
    for y in range(len(ticker_prices)):
        adj_close.append(ticker_prices['Adj Close'][y])
        date.append(ticker_prices['Date'][y])

    return ticker_select,date,adj_close
