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
#nltk.download('stopwords')
#nltk.download('punkt')

from pathlib import Path
import base64

@st.cache_data
def nltk_():
    nltk.download('stopwords')
    nltk.download('punkt')



@st.cache_data
def get_comments(link):
    #os.system(f"youtube-comment-downloader --url {link} --output scraping.txt")
    comments = downloader.get_comments_from_url(youtube_url=link, sort_by=SORT_BY_POPULAR)
    comments_list = list(comments)
    return comments_list


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
        print("processing is finished")
        return df_pr

st.set_page_config(
     page_title='YOUTUBE COMMENT ANALYSIS',
     layout="wide",
     initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/akdilali',
        'Report a bug': "https://github.com/akdilali",
        'About': "# This is a training app!"
    }
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def dasboard(video_link):
    if video_link !='':
        print("link is entered")
        st.sidebar.write(':blue[Data is collecting...]')
        comments = get_comments(video_link)
        st.sidebar.write(":blue[Data is collected. Analyze is started...]")
        
        Pr =Process()
        
        df_len = Pr.comments_length(comments)
        df_len.sort_values(by='length',ascending=False, inplace=True)
        dict_len = { "index":["minimum","maximum","average"],
                "length of comments": [int(df_len.length.min()),int(df_len.length.max()),int(df_len.length.mean())]}

        df_length = pd.DataFrame(dict_len)
        df_length.set_index('index', inplace=True)
        st.header(":blue[Comments length]")
        st.table(df_length)        

        df = Pr.processing(comments)

        top_users_df = (
        df[['author','votes']].groupby("author")
        .count()
        .reset_index()
        .rename(columns={"votes": "Count of Comments"})
        .sort_values(by="Count of Comments", ascending=False)
        )
        fig_top_user = px.bar(
            top_users_df.head(10),
            x="author",
            y="Count of Comments",
            color="author",
            text="Count of Comments",
        )
        st.header(":blue[Most commented users]")
        st.plotly_chart(fig_top_user,use_container_width=True)

        df_most_words = Pr.count_words_fast(df)
        df_most_words.sort_values(by='Count', ascending=False, inplace=True)

        st.header(":blue[Most repeated words]")
        fig_most_words = px.bar(
            df_most_words,
            x="Word",
            y="Count",
            color="Word",
            text="Count",
        )
        st.plotly_chart(fig_most_words,use_container_width=True) 
        st.sidebar.write(":blue[Analysis continues. \n You can examine the completed graphs. \nPlease wait a few more minutes]")

        @st.cache_resource
        def load_emotion():
            return create_analyzer(task="emotion", lang="en")
        
        @st.cache_resource
        def load_sentiment():
            return create_analyzer(task="sentiment", lang="en")

        @st.cache_resource
        def load_hate_speech():
            return create_analyzer(task="hate_speech", lang="en")
        
        emotion_analyzer = load_emotion()
        sentiment_analyzer = load_sentiment()
        hate_speech = load_hate_speech()
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
#        df['aggressive_rate'] = ag
#        df['targeted_rate'] = trg

        sentiment_df = (
            df[['sentiment of comments','votes']].groupby("sentiment of comments")
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
            title='Positivity, negativity and neutral status of comments'
        )
        st.header(":blue[Comments sentiment]")
        st.plotly_chart(fig_sentiment,use_container_width=True)

        emotion_df = (
            df[['emotion of comments','votes']].groupby("emotion of comments")
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
        st.header(":blue[Emotions of comments]")
        st.plotly_chart(fig_emotion, use_container_width=True)

        sentiment_emotion_df = (
            df[['sentiment of comments','emotion of comments','votes']].groupby(['sentiment of comments','emotion of comments'])
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
            title='Comment emotions with positivity, negativity and neutral status of comments'

        )
        st.header(":blue[comments emotions with sentiment]")
        st.plotly_chart(fig_sentiment_emotion, use_container_width=True)
        #yüzdeye çevrilecek
        dict_sentiment = { "index":["minimum","maximum","average"],
                "hate speech rate": [f"%{round(df.hate_speech_rate.min()*100,2)}",f"%{round(df.hate_speech_rate.max()*100,2)}",f"%{round(df.hate_speech_rate.mean()*100,2)}"]}
#                "aggressive speech rate": [df.aggressive_rate.min(), df.aggressive_rate.max(), df.aggressive_rate.mean()], 
#                "targeted speech rate":[df.targeted_rate.min(),df.targeted_rate.max(),df.targeted_rate.mean()]}

        best_df = df[(df['emotion of comments'] == 'joy') & (df['sentiment of comments'] == 'POS')]
        best_df.sort_values(by='hate_speech_rate',ascending=True, inplace=True)
        best_df.reset_index(inplace=True)
        worst_df = df[(df['emotion of comments'] == 'disgust') & (df['sentiment of comments'] == 'NEG')]
        worst_df.sort_values(by='hate_speech_rate',ascending=False, inplace=True)
        worst_df.reset_index(inplace=True)
        df_sentiment = pd.DataFrame(dict_sentiment)
        df_sentiment.set_index('index', inplace=True)
        st.header(":blue[Bad comments rate]")
        st.table(df_sentiment)
        st.header(":blue[Best comments]")
        best_df = best_df[['text','author']][0:5]
        st.markdown(best_df.style.hide(axis="index").to_html(),unsafe_allow_html=True)
        st.header(":blue[Worst comments]")
        worst_df = worst_df[['text','author']][0:5]
        st.markdown(worst_df.style.hide(axis="index").to_html(),unsafe_allow_html=True)
    return None

def main():

    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=75 height=50>](https://streamlit.io/)'''.format(img_to_bytes("icon.png")), unsafe_allow_html=True)
    st.sidebar.header(':blue[YOUTUBE COMMENT ANALYSIS]', divider='blue')
    
    st.sidebar.write("Language should be english")
    video_link = st.sidebar.text_input(":blue[paste the link below for analysis]",placeholder='PASTE LINK')
    #st.sidebar.write(':blue[VIDEO LINK:] ', video_link)

    st.sidebar.write("Project Repository: [link](https://github.com/akdilali/youtube-comment-analysis/)")

    

    if st.sidebar.button(':blue[START ANALYSIS]'):
        st.sidebar.write(':blue[VIDEO LINK:] ', video_link)
        start_time = time.time()
        st.markdown("<h1 style='text-align: center; color: blue;'>YouTube Comment Analysis Dashboard</h1>", unsafe_allow_html=True)
        dasboard(video_link)
        st.sidebar.write(":blue[Analysis finished]")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")

# Run main()

if __name__ == '__main__':
    nltk_()
    main()
