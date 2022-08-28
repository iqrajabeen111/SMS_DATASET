# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import nltk
import string
import re
from collections import Counter
import matplotlib.pyplot as plt
#ALL REQUIRED LIBRARIES FROM NLTK TO PROCESS DATA

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')



from nltk.corpus import stopwords
", ".join(stopwords.words('english'))



import plotly.graph_objects as go



def remove_urls(text):
    url_pattern = re.compile(r'[0-9]+')
    return url_pattern.sub('', text)

def remove_numbers(text):
    url_pattern = re.compile(r'http\S+')
    return url_pattern.sub('', text)

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



def Word_Count(data_frame_col,counter_obj):
  for text in data_frame_col.values:
    for word in text.split():
      counter_obj[word] += 1
  return counter_obj


def main():
    st.title("Dataset for SMS")
    path = 'C:/Users/iqrajabeen/pythonclass/class5/'
    filename ='SMS_data.csv'
    complete_filepath = filename 
    
    full_df = pd.read_csv(complete_filepath, encoding= 'unicode_escape' )
  
    
  #PERFORM DATA CLEANING CHECKS#
  #check if their any null or missing value in data
    full_df.isna().sum()
    #drop if any nan of missing value is in the data and compare your dropeddata with original data
    dropped_data = full_df.dropna()
    # print(dropped_data.shape)
    full_df.equals(dropped_data)
    #check if any duplicate value exist we can not remove data because there can be same data of sms
    #check data type of columns

    
    # temp = full_df['Date_Received'].replace('1/1/1900', '')
    # converted_datetime = pd.to_datetime(temp)
    # full_df['Date_Received'] = converted_datetime
    #print(full_df.info())
  
    
    Label=st.selectbox('Select',full_df['Label'].unique())
    button=st.button('show result')
    if button:
        subset=full_df[full_df['Label']==Label]
        

        full_df["text_processed"] = (
        
        full_df["Message_body"].str.lower() #LOWER CASED EVERYTHING
    
        .apply(lambda text: remove_urls(text)) #URLS REMOVED
    
        .apply(lambda text: remove_numbers(text)) #NUMBERS REMOVED
    
        .apply(lambda text: remove_punctuation(text)) #REMOVE PUNCTUATION
    
        .apply(lambda text: remove_stopwords(text)) #REMOVED STOP WORDS 
    
        .apply(lambda text: lemmatize_words(text)) #LEMMATIZED TO ROOT WORD
    
        )
        
        
    
        cnt_words_dict_inbound=Word_Count(full_df.query("Label == @Label")['text_processed'],Counter())
        a=cnt_words_dict_inbound.most_common(10)
        data = pd.DataFrame(a)
        
        #data.columns = ['words','count']
            
        #total_words = data['words'].values
        #total_count = data['count'].values
        figu=plt.figure(figsize=(5,2))
        
        sns.barplot(x=data[1],y=data[0])
        plt.gca().set(title="common words",xlabel="word-count")
        st.pyplot(figu)
        
        #st.bar_chart(data['words'])
    
            #seperate spam and non spam values
        spam_data=full_df[full_df['Label'] == Label]
        
        #print(full_df.info())
        
        getdata=spam_data.groupby(['Date_Received'])['Message_body'].count()
        st.line_chart(getdata)
  
       




if __name__ == '__main__':
    main()