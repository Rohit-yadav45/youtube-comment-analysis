from itertools import islice
from youtube_comment_downloader import *
downloader = YoutubeCommentDownloader()

import streamlit as st
import pandas as pd
import plotly.express as px

import os
import time
import json

import re
from collections import Counter
from pysentimiento import create_analyzer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from pathlib import Path
import base64


class Scraping:
    def __init__(self,link):
        self.link=link

    def get_comments(self):
        os.system(f"youtube-comment-downloader --url {self.link} --output scraping.txt")

def comment_liste():
    ufc_comments_list=[]
    # Step 1: Open the text file in read mode
    with open('scraping.txt', 'r', encoding="utf8") as file:
        for line in file:
            # Step 2: Parse each line as a JSON object
            data = json.loads(line)
            ufc_comments_list.append(data)
    print("comment list is taken")
    return ufc_comments_list


class Process:

    def remove_not_str(self,text):
        text = text.lower()
        letters_with_spaces = re.findall(r'[a-zA-Z\s]+', text)
        result = ''.join(letters_with_spaces)
        return result

    def count_words_fast(self,processed_df):
        text = ''
        for x in processed_df['text']: text+=x
        text = text.lower()
        letters_with_spaces = re.findall(r'[a-zA-Z\s]+', text)
        result = ''.join(letters_with_spaces)
        word_counts = Counter(result.split(" "))
        sorted_dict = dict(sorted(word_counts.items(), key=lambda item: item[1]))
        first_ten_elements = {key: sorted_dict[key] for key in list(sorted_dict)[-10:]}
        df_word_count = pd.DataFrame(list(first_ten_elements.items()), columns=['Word', 'Count'])
        return df_word_count
    
    def comments_length(self,comment_liste):
        d = {'comment': [x['text'] for x in comment_liste]}
        df_len = pd.DataFrame(data=d)
        df_len.dropna(inplace=True)
        df_len['length'] = df_len['comment'].apply(lambda x: len(x))
        return df_len
    
    def processing(self,comment_liste):
        d = {'cid': [x['cid'] for x in comment_liste], 'text': [x['text'] for x in comment_liste],
            'time': [x['time'] for x in comment_liste],'author': [x['author'] for x in comment_liste],
            'votes': [x['votes'] for x in comment_liste]}
        df_pr = pd.DataFrame(data=d)
        df_pr.dropna(inplace=True)
        print("dataframe is created")
        df_pr['text'] = df_pr['text'].apply(self.remove_not_str)
        print("words is cleaned")
        df_pr['text'] = df_pr['text'].apply(lambda x: ' '.join([word for word in nltk.word_tokenize(x) if word.lower() not in set(stopwords.words('english'))]) )
        print("stop words was removed")
        #df_pr['text'] = df_pr['text'].apply(lambda x: str(TextBlob(x).correct()))
        #print("words is correct now")        
        print("processing is finished")
        #df_pr.to_csv("new_df55555.csv")
        return df_pr

st.set_page_config(
     page_title='YOUTUBE COMMENT ANALYSIS',
     layout="wide",
     initial_sidebar_state="expanded",
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def dasboard(video_link):
    if video_link !='':
        print("link is entered") 
        sc = Scraping(video_link)
        sc.get_comments()
        comments = comment_liste()
        time.sleep(5)
        st.write(":blue[Data is collected. Analyze is started...]")
        Pr =Process()
        
        df_len = Pr.comments_length(comments)
        df_len.sort_values(by='length',ascending=False, inplace=True)
        dict_len = { "index":["min","max","median"],
                "length": [int(df_len.length.min()),int(df_len.length.max()),int(df_len.length.mean())]}

        df_length = pd.DataFrame(dict_len)
        #df_length.set_index('index', inplace=True)
        st.title(":blue[COMMENTS LENGTH]")
        st.table(df_length)        

        df = Pr.processing(comments)
        selected_columns = ['author','votes']
        temp_df = df[selected_columns]

        top_users_df = (
        temp_df.groupby("author")
        .count()
        .reset_index()
        .rename(columns={"votes": "Count"})
        .sort_values(by="Count", ascending=False)
        )
        fig_top_user = px.bar(
            top_users_df.head(10),
            x="author",
            y="Count",
            color="author",
            text="Count",
        )
        st.title(":blue[MOST COMMENTED USERS]")
        st.plotly_chart(fig_top_user)

        df_most_words = Pr.count_words_fast(df)
        df_most_words.sort_values(by='Count', ascending=False, inplace=True)
        st.title(":blue[MOST REPEATED WORDS]")
        st.table(df_most_words)
        fig_most_words = px.bar(
            df_most_words,
            x="Word",
            y="Count",
            color="Word",
            text="Count",
        )
        st.plotly_chart(fig_most_words)     

        emotion_analyzer = create_analyzer(task="emotion", lang="en")
        sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
        hate_speech = create_analyzer(task="hate_speech", lang="en")
        emo = []
        sen = []
        ht= []
        ag=[]
        trg=[]
        for i in df['text']:
            out_sen = sentiment_analyzer.predict(i)
            out_emo = emotion_analyzer.predict(i)
            out_ht = hate_speech.predict(i)
            sen.append(out_sen.output)
            emo.append(out_emo.output)
            ht.append(out_ht.probas['hateful'])
            ag.append(out_ht.probas['aggressive'])
            trg.append(out_ht.probas['targeted'])

        df['sentiment of comments'] = sen
        df['emotion of comments'] = emo
        df['hate_speech_rate'] = ht
        df['aggressive_rate'] = ag
        df['targeted_rate'] = trg

        selected_columns = ['sentiment of comments','votes']
        temp_df = df[selected_columns]
        sentiment_df = (
            temp_df.groupby("sentiment of comments")
            .count()
            .reset_index()
            .rename(columns={"votes": "Count"})
            .sort_values(by="Count", ascending=False)
        )
        fig_sentiment = px.bar(
            sentiment_df,
            x="sentiment of comments",
            y="Count",
            color=["NOTR","NEGATIVE","POSITIVE"],
            text="Count",
        )
        st.title(":blue[COMMENTS SENTIMENT]")
        st.plotly_chart(fig_sentiment)

        selected_columns = ['emotion of comments','votes']
        temp_df = df[selected_columns]
        emotion_df = (
            temp_df.groupby("emotion of comments")
            .count()
            .reset_index()
            .rename(columns={"votes": "Count"})
            .sort_values(by="Count", ascending=False)
        )
        fig_emotion = px.bar(
            emotion_df,
            x="emotion of comments",
            y="Count",
            color="emotion of comments",
            text="Count",
        )
        st.title(":blue[COMMENTS EMOTION]")
        st.plotly_chart(fig_emotion)

        selected_columns = ['sentiment of comments','emotion of comments','votes']
        temp_df = df[selected_columns]
        sentiment_emotion_df = (
            temp_df.groupby(['sentiment of comments','emotion of comments'])
            .count()
            .reset_index()
            .rename(columns={"votes": "Count"})
            .sort_values(by="Count", ascending=False)
        )
        fig_sentiment_emotion = px.bar(
            sentiment_emotion_df,
            x="sentiment of comments",
            y="Count",
            color='emotion of comments',
            text="Count",
        )
        st.title(":blue[COMMENTS SENTIMENT WITH EMOTION]")
        st.plotly_chart(fig_sentiment_emotion)
        #yüzdeye çevrilecek
        dict_sentiment = { "index":["min","max","median"],
                "hate_Speech": [df.hate_speech_rate.min(),df.hate_speech_rate.max(),df.hate_speech_rate.mean()] ,
                "aggressive": [df.aggressive_rate.min(), df.aggressive_rate.max(), df.aggressive_rate.mean()], 
                "targeted":[df.targeted_rate.min(),df.targeted_rate.max(),df.targeted_rate.mean()]}

        best_df = df[(df['emotion of comments'] == 'joy') & (df['sentiment of comments'] == 'POS')]
        best_df.sort_values(by=['hate_speech_rate','aggressive_rate','targeted_rate'],ascending=True, inplace=True)
        best_df.reset_index(inplace=True)
        worst_df = df[(df['emotion of comments'] == 'disgust') & (df['sentiment of comments'] == 'NEG')]
        worst_df.sort_values(by=['hate_speech_rate','aggressive_rate','targeted_rate'],ascending=False, inplace=True)
        worst_df.reset_index(inplace=True)
        df_sentiment = pd.DataFrame(dict_sentiment)
        df_sentiment.set_index('index', inplace=True)
        st.title(":blue[BAD COMMENTS RATE]")
        st.table(df_sentiment)
        st.title(":blue[BEST COMMENTS]")
        st.table(best_df[['text','author']].head())
        st.title(":blue[WORST COMMENTS]")
        st.table(worst_df[['text','author']].head())
    return None

def main():

    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://streamlit.io/)'''.format(img_to_bytes("icon.png")), unsafe_allow_html=True)
    st.sidebar.header(':blue[YOUTUBE COMMENT ANALYSIS DASHBOARD]', divider='blue')
    
    st.sidebar.markdown('''
                        <small>Project Repostory: [github](https://github.com/akdilali/).</small>
                        ''', unsafe_allow_html=True)

    video_link = st.sidebar.text_input(":blue[INPUT LINK BELOW THE FILED FOR ANALYZE]",placeholder='PASTE LINK')
    st.sidebar.write(':blue[VIDEO LINK:] ', video_link)

    if st.sidebar.button(':blue[START ANALYZE]'):
        dasboard(video_link)

# Run main()

if __name__ == '__main__':
    main()