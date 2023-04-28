#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from prophet import Prophet
import folium
import httpx
import requests, json, csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from string import punctuation

import stopwords as stopwords
import nltk
from wordcloud import WordCloud


# In[15]:


# Load the 2019_Q1 data
bike_2019_Q1 = pd.read_csv("C:\DATA\metro-bike-share-trips-2019-q1.csv", index_col=2, parse_dates=[2,3])
bike_2019_Q1=bike_2019_Q1.fillna(0)
bike_2019_Q1.head()


# In[16]:


bike_2019_Q1.dtypes


# In[17]:


# Load the 2019_Q2 data
bike_2019_Q2 = pd.read_csv("C:\DATA\metro-bike-share-trips-2019-q2.csv", index_col=2, parse_dates=[2,3])
bike_2019_Q2=bike_2019_Q2.fillna(0)
bike_2019_Q2.head()


# In[18]:


# merge the two data frames
print('After merging:')
bike_2019_H1=pd.concat([bike_2019_Q1, bike_2019_Q2], axis=0)
bike_2019_H1.head()


# In[19]:


# Load the 2019_Q3 data
bike_2019_Q3 = pd.read_csv("C:\DATA\metro-bike-share-trips-2019-q3.csv", index_col=2, parse_dates=[2,3])
bike_2019_Q3=bike_2019_Q3.fillna(0)
bike_2019_Q3.head()


# In[21]:


#Check for missing values 
bike_2019_Q3.isna().sum()


# In[22]:


# Total number of records
len(bike_2019_Q3)


# In[23]:


# Load the 2019_Q4 data
bike_2019_Q4 = pd.read_csv("C:\DATA\metro-bike-share-trips-2019-q4.csv", index_col=2, parse_dates=[2,3])
bike_2019_Q4=bike_2019_Q4.fillna(0)
bike_2019_Q4.head()


# In[24]:


# Total number of records
len(bike_2019_Q4)


# In[25]:


# merge the two data frames
print('After merging:')
bike_2019_H2=pd.concat([bike_2019_Q3, bike_2019_Q4], axis=0)
bike_2019_H2.head()


# In[26]:


# Merge all the data frames
print('After merging:')
bike_2019=pd.concat([bike_2019_H1, bike_2019_H2], axis=0)
bike_2019.head()
bike_2019.tail()


# In[27]:


# Total number of records
len(bike_2019)


# In[28]:


# Checking for the datatypes of the data frame
bike_2019.dtypes


# In[29]:


#Check for missing values 
bike_2019.isna().sum()


# # Prophet

# In[30]:


bike_2019 = bike_2019.reset_index()
weekly_trips = bike_2019.resample('W', on='start_time')['trip_id'].count()


# In[31]:


weekly_trips_2019 = weekly_trips[weekly_trips.index.year == 2019]


# In[32]:


prophet_data = pd.DataFrame()
prophet_data['ds'] = weekly_trips_2019.index
prophet_data['y'] = weekly_trips_2019.values


# In[33]:


model = Prophet()
model.fit(prophet_data)


# In[34]:


future = model.make_future_dataframe(periods=52, freq='W')
forecast = model.predict(future)


# In[35]:


next_year_forecast = forecast[forecast['ds'].dt.year == 2020]
total_trips_next_year = next_year_forecast['yhat'].sum()
print(f"Total predicted bike trips for the next year: {total_trips_next_year}")


# In[36]:


fig = model.plot(forecast)
plt.title('Weekly Bike Trips with Prophet Predictions')
plt.xlabel('Week')
plt.ylabel('Number of Trips')
plt.show()


# # 2020

# In[37]:


# Load the 2020_Q1 data
bike_2020_Q1 = pd.read_csv("C:\DATA\metro-bike-share-trips-2020-q1.csv", index_col=2, parse_dates=[2,3])
bike_2020_Q1=bike_2020_Q1.fillna(0)
bike_2020_Q1.head()


# In[38]:


# Load the 2020_Q2 data
bike_2020_Q2 = pd.read_csv("C:\DATA\metro-trips-2020-q2-v2.csv", index_col=2, parse_dates=[2,3])
bike_2020_Q2=bike_2020_Q2.fillna(0)
bike_2020_Q2.head()


# In[39]:


# merge the two data frames
print('After merging:')
bike_2020_H1=pd.concat([bike_2020_Q1, bike_2020_Q2], axis=0)
bike_2020_H1.head()


# In[40]:


# Load the 2020_Q3 data
bike_2020_Q3 = pd.read_csv("C:\DATA\metro-trips-2020-q3.csv", index_col=2, parse_dates=[2,3])
bike_2020_Q3=bike_2020_Q3.fillna(0)
bike_2020_Q3.head()


# In[41]:


# Load the 2020_Q4 data
bike_2020_Q4 = pd.read_csv("C:\DATA\metro-trips-2020-q4.csv", index_col=2, parse_dates=[2,3])
bike_2020_Q4=bike_2020_Q4.fillna(0)
bike_2020_Q4.head()


# In[42]:


# merge the two data frames
print('After merging:')
bike_2020_H2=pd.concat([bike_2020_Q3, bike_2020_Q4], axis=0)
bike_2020_H2.head()


# In[43]:


# merge all the data frames
print('After merging:')
bike_2020=pd.concat([bike_2020_H1, bike_2020_H2], axis=0)
bike_2020.head()


# In[44]:


# Total number of records
len(bike_2020)


# In[45]:


# Checking for the datatypes of the data frame
bike_2020.dtypes


# In[46]:


bike_2020 = bike_2020.reset_index()
weekly_trips = bike_2020.resample('W', on='start_time')['trip_id'].count()


# In[47]:


weekly_trips_2020 = weekly_trips[weekly_trips.index.year == 2020]


# In[48]:


weekly_trips_forecast = forecast['yhat'].iloc[-52:].values


# In[49]:


# Get the week numbers for 2019 and 2020
week_numbers_2019 = weekly_trips_2019.index.week
week_numbers_2020 = weekly_trips_2020.index.week

# Create the x-axis values for the forecast
forecast_week_numbers = np.arange(1, 53)

# Plot the data from 2019, 2020, and the forecasted values
plt.figure(figsize=(14, 6))
plt.plot(week_numbers_2019, weekly_trips_2019.values, label="2019", color='blue')
plt.plot(week_numbers_2020, weekly_trips_2020.values, label="2020", color='green')
plt.plot(forecast_week_numbers, weekly_trips_forecast, label="Forecast_2020", color='red')
plt.xlabel('Week')
plt.ylabel('Trips')
plt.title('Weekly Trips (2019, 2020, and Forecast based on 2019)')
plt.legend(loc='best')
plt.show()


# # 2019 map Green is most popular, and red is least
# 

# In[50]:


bike_2019['start_station_Count'] = bike_2019.groupby('start_station')['start_station'].transform('count')
bike_2019.head()


# In[51]:


bike_2019=bike_2019[['start_station', 'start_lat', 'start_lon', 'start_station_Count']]
print(bike_2019.head())
print(len(bike_2019))


# In[52]:


# Using DataFrame.drop_duplicates() to keep first duplicate row
bike_2019 = bike_2019.drop_duplicates(keep='first')
print(len(bike_2019))


# In[55]:


# Sort the DataFrame by counts in descending order
bike_2019 = bike_2019.sort_values('start_station_Count', ascending=False)
#Trim the dataframe to utmost 50 rows

# Get the station with the highest counts and lowest 
top_20_stations = bike_2019.iloc[:20]
least_20_stations = bike_2019.iloc[-20:]

# Create a folium map centered on the station with the highest counts
map = folium.Map(location=[bike_2019.iloc[0]['start_lat'], bike_2019.iloc[0]['start_lon']], zoom_start=12)

# Add markers for each station on the map
for index, row in bike_2019.iterrows():
    if row['start_station'] in top_5_stations['start_station'].values:
        color = 'green'
    elif row['start_station'] in least_5_stations['start_station'].values:
        color = 'red'
    else:
        color = 'blue'
    folium.Marker(
        location=[row['start_lat'], row['start_lon']],
        popup=f"Station: {row['start_station']}<br>Counts: {row['start_station_Count']}",
        icon=folium.Icon(color=color)
    ).add_to(map)

# Display the map
map


# In[57]:


print(top_20_stations)


# In[58]:


print(least_20_stations)


# # NLP

# ## get review from twitter

# In[2]:


company_name = 'Metro Bike'
data_save_path = 'd:/课件/tmpcsv/' + company_name+ '_tweet.csv'
emotion_data_save_path = 'd:/课件/tmpcsv/' + company_name+ '_tweet_emotion.csv'
pos_keyword_save_path = 'd:/课件/tmpcsv/' + company_name+ '_pos_key.csv'
neg_keyword_save_path = 'd:/课件/tmpcsv/' + company_name+ '_neg_key.csv'
data_num = 600


# In[3]:


cookies = {
    '_gid': 'GA1.2.846350087.1681798648',
    'at_check': 'true',
    '_ga_BYKEBDM7DS': 'GS1.1.1681799009.1.1.1681799896.0.0.0',
    'lang': 'en',
    'des_opt_in': 'Y',
    '_gcl_au': '1.1.1784331066.1681803769',
    'mbox': 'PC#5b02907c721345758f5225fa4b5991bb.38_0#1745048661|session#ac62135c02ea4f038a020f3515cafc3b#1681805721',
    '_ga_34PHSZMC42': 'GS1.1.1681802494.1.1.1681803869.0.0.0',
    '_ga': 'GA1.2.2091119673.1681798648',
    'g_state': '{"i_l":2,"i_p":1681912775160}',
    'guest_id_ads': 'v1%3A168191479855080063',
    'guest_id_marketing': 'v1%3A168191479855080063',
    'gt': '1648696277759311877',
    'guest_id': 'v1%3A168191479855080063',
    'personalization_id': '"v1_SKsjxWm6m9RRJN9AoR9Dyg=="',
    'ct0': 'ebbb2d6c39c472c41673801ca578da2a',
}


# In[4]:


url_token = 'https://api.twitter.com/1.1/guest/activate.json'


# In[5]:


def get_token():
    token = json.loads(requests.post(url_token, headers=headers).text)['guest_token']
    headers['x-guest-token'] = token


# In[6]:


headers = {
    'authority': 'twitter.com',
    'accept': '*/*',
    'accept-language': 'zh-CN,zh;q=0.9',
    'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
    # 'cookie': '_gid=GA1.2.846350087.1681798648; at_check=true; _ga_BYKEBDM7DS=GS1.1.1681799009.1.1.1681799896.0.0.0; lang=en; des_opt_in=Y; _gcl_au=1.1.1784331066.1681803769; mbox=PC#5b02907c721345758f5225fa4b5991bb.38_0#1745048661|session#ac62135c02ea4f038a020f3515cafc3b#1681805721; _ga_34PHSZMC42=GS1.1.1681802494.1.1.1681803869.0.0.0; _ga=GA1.2.2091119673.1681798648; guest_id_ads=v1%3A168180559609265290; guest_id_marketing=v1%3A168180559609265290; guest_id=v1%3A168180559609265290; external_referer=padhuUp37zjvUL3S7Q36NDDCPvFZ8go1ASlFUACs362gy2wBjI7M2tI1nyf6XEbyuZhjxi%2Fn1f67L4anGIuxdPZ0x0HqqP7r%2FedNmEd00q8%3D|0|8e8t2xd8A2w%3D; g_state={"i_l":2,"i_p":1681912775160}; gt=1648330264706416640; att=1-kLwPPqww5ViJmrc6NymMB8KPAwFb5XFaepkVXCvW; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCP%252FL1pSHAToMY3NyZl9p%250AZCIlNjI4YzJlNGQ0OGYwMDQ1YjVjMTUyMGNhYjNjZGU0ZTE6B2lkIiVhMTA1%250AMTdjZTU3MmUxYTcxZDE0NWE3M2RlOGY1N2QyZA%253D%253D--285aee7b0c1f5234f3cef4d19bd3bd803eb87811; personalization_id="v1_GvZYH8cT1m+HIurtNlzIeA=="; ct0=6fc7c4740ba6c19bf1d6d9b4681389b9',
    'referer': 'https://twitter.com/search?q=citibike&src=typed_query&f=live',
    'sec-ch-ua': '"Google Chrome";v="111", "Not(A:Brand";v="8", "Chromium";v="111"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36',
    'x-csrf-token': '6fc7c4740ba6c19bf1d6d9b4681389b9',
    # 'x-guest-token': '1648330264706416640',
    'x-twitter-active-user': 'yes',
    'x-twitter-client-language': 'zh-cn',
}


# In[7]:


params = {
    'include_profile_interstitial_type': '1',
    'include_blocking': '1',
    'include_blocked_by': '1',
    'include_followed_by': '1',
    'include_want_retweets': '1',
    'include_mute_edge': '1',
    'include_can_dm': '1',
    'include_can_media_tag': '1',
    'include_ext_has_nft_avatar': '1',
    'include_ext_is_blue_verified': '1',
    'include_ext_verified_type': '1',
    'include_ext_profile_image_shape': '1',
    'skip_status': '1',
    'cards_platform': 'Web-12',
    'include_cards': '1',
    'include_ext_alt_text': 'true',
    'include_ext_limited_action_results': 'false',
    'include_quote_count': 'true',
    'include_reply_count': '1',
    'tweet_mode': 'extended',
    'include_ext_views': 'true',
    'include_entities': 'true',
    'include_user_entities': 'true',
    'include_ext_media_color': 'true',
    'include_ext_media_availability': 'true',
    'include_ext_sensitive_media_warning': 'true',
    'include_ext_trusted_friends_metadata': 'true',
    'send_error_codes': 'true',
    'simple_quoted_tweet': 'true',
    'q': company_name,
    'tweet_search_mode': 'live',
    'query_source': 'typed_query',
    'count': '50',
    'requestContext': 'launch',
    'pc': '1',
    'spelling_corrections': '1',
    'include_ext_edit_control': 'true',
    'ext': 'mediaStats,highlightedLabel,hasNftAvatar,voiceInfo,enrichments,superFollowMetadata,unmentionInfo,editControl,vibe',
}


# In[8]:


get_token()

cursur_value = ''

def getParam(curs):
    return {
    'include_profile_interstitial_type': '1',
    'include_blocking': '1',
    'include_blocked_by': '1',
    'include_followed_by': '1',
    'include_want_retweets': '1',
    'include_mute_edge': '1',
    'include_can_dm': '1',
    'include_can_media_tag': '1',
    'include_ext_has_nft_avatar': '1',
    'include_ext_is_blue_verified': '1',
    'include_ext_verified_type': '1',
    'include_ext_profile_image_shape': '1',
    'skip_status': '1',
    'cards_platform': 'Web-12',
    'include_cards': '1',
    'include_ext_alt_text': 'true',
    'include_ext_limited_action_results': 'false',
    'include_quote_count': 'true',
    'include_reply_count': '1',
    'tweet_mode': 'extended',
    'include_ext_views': 'true',
    'include_entities': 'true',
    'include_user_entities': 'true',
    'include_ext_media_color': 'true',
    'include_ext_media_availability': 'true',
    'include_ext_sensitive_media_warning': 'true',
    'include_ext_trusted_friends_metadata': 'true',
    'send_error_codes': 'true',
    'simple_quoted_tweet': 'true',
    'q': company_name,
    'tweet_search_mode': 'live',
    'count': '20',
    'query_source': 'typed_query',
    'cursor': curs,
    'pc': '1',
    'spelling_corrections': '1',
    'include_ext_edit_control': 'true',
    'ext': 'mediaStats,highlightedLabel,hasNftAvatar,voiceInfo,enrichments,superFollowMetadata,unmentionInfo,editControl,vibe',
    }


# In[9]:


data_list = []
cnt = 0
page = 1
url = 'https://twitter.com/i/api/2/search/adaptive.json'


# In[ ]:


while cnt < data_num :
    response = None
    try:
        if page == 1:
            response = httpx.get(url, params=params, cookies=cookies,
                                 headers=headers)
        else:
            response = httpx.get(url, params=getParam(cursur_value),
                                 cookies=cookies,
                                 headers=headers)
    except Exception:
        print('网络错误')
        continue


    # print(response.text)
    if response.status_code == 200:
        res_json = json.loads(response.text)
        tweets = res_json['globalObjects']['tweets']
        for value in tweets.values():
            # print(value['full_text'])


            if '线路' in value['full_text'] or 'vote' in value['full_text'] or 'voting' in value['full_text'] or len(value['full_text']) < 42:
                continue
            cnt = cnt + 1

            data_list.append((cnt,  value['full_text'].replace('\s\n', '')  ))
            # saveTweet(value['full_text'])

        entries = res_json['timeline']['instructions'][0]['addEntries']['entries']
        cursur_exist = False
        for entry in entries:
            if entry['entryId'] == 'sq-cursor-bottom':
                cursur = entry['content']['operation']['cursor']['value']
                cursur_value = cursur
                cursur_exist = True
                print(cursur_value)
        if not cursur_exist:
            entry = res_json['timeline']['instructions'][-1]['replaceEntry']['entry']
            if entry['entryId'] == 'sq-cursor-bottom':
                cursur = entry['content']['operation']['cursor']['value']
                cursur_value = cursur
                cursur_exist = True
                print(cursur_value)


        page = page + 1


# In[ ]:


def saveDatas(lines):
    with open(data_save_path, 'w',  encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(lines)


# In[ ]:


saveDatas(data_list)


# ## emotional analysis

# In[4]:


#nltk.download('stopwords')

def removeStopWords(sentence):
    # Input: a sentence of tweet
    # Output: the sentence of input tweet, but non-stop + non-punctuation words are removed
    stop_words = set(stopwords.get_stopwords('english'))
    tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in tokens if not w.lower() in stop_words]
    removed_punct_sentence = [w for w in filtered_sentence if not w in punctuation]
    return removed_punct_sentence


# In[5]:


input_file = open(data_save_path, encoding='utf8')
output_file = open(emotion_data_save_path, mode="w",encoding='utf8',  newline='')
csv_writer = csv.writer(output_file)
analyzer = SentimentIntensityAnalyzer()
lines = csv.reader(input_file, skipinitialspace= True)


pos_list = []
neg_list = []


# In[6]:


for line in lines:
    # ind = line.find(',')
    # line_num = line[0:ind]
    # sentence = line[ind+1:]
    line_num = line[0]
    sentence = line[1]
    vs = analyzer.polarity_scores(sentence)


    eval = ''
    if vs['compound'] > 0.05:
        pos_list.extend(removeStopWords(sentence))
        eval = 'positive'
    elif vs['compound'] <-0.05:
        neg_list.extend(removeStopWords(sentence))
        eval = 'negative'
    else:
        eval = 'neutral'
    csv_writer.writerow([line_num, sentence, eval])
    # output_file.write(str(sentence)+str(vs['compound'])+'\n')

input_file.close()
output_file.close()


# In[7]:


sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
with open(emotion_data_save_path, encoding='utf8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        sentiment = row[2]
        sentiment_counts[sentiment] += 1


# In[8]:


labels = ['Positive', 'Negative', 'Neutral']
sizes = [sentiment_counts['positive'], sentiment_counts['negative'], sentiment_counts['neutral']]
colors = ['#99ff99', '#ff9999', '#66b3ff']
explode = (0.1, 0.1, 0.1)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Ensures that the pie chart is circular
plt.title("Sentiment Distribution")
plt.show()


# In[10]:


output_file = open(pos_keyword_save_path, mode="w",encoding='utf8',  newline='')
csv_writer = csv.writer(output_file)
counter = Counter(pos_list)
keyworkd_list = counter.most_common(100)
csv_writer.writerows(keyworkd_list)
output_file.close()


# In[11]:


output_file = open(neg_keyword_save_path, mode="w",encoding='utf8',  newline='')
csv_writer = csv.writer(output_file)
counter = Counter(neg_list)
keyworkd_list = counter.most_common(100)
csv_writer.writerows(keyworkd_list)
output_file.close()


# In[13]:



# Load positive and negative keywords from files
with open(pos_keyword_save_path, mode='r', encoding='utf8') as file:
    pos_words = file.read().splitlines()

with open(neg_keyword_save_path, mode='r', encoding='utf8') as file:
    neg_words = file.read().splitlines()

# Combine the two lists
all_words = pos_words + neg_words

# Generate word cloud
wordcloud = WordCloud(width=800, height=200, background_color='white', max_words=100).generate(' '.join(all_words))

# Plot the word cloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




