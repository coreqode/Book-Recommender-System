#!/usr/bin/env python
# coding: utf-8

# In[841]:


import os
import sys
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('books.csv')
tags = pd.read_csv('tags.csv')

book_tags = pd.read_csv('book_tags.csv')

tags = pd.merge( book_tags,tags, left_on = 'tag_id', right_on = 'tag_id', how = 'inner')

merged_books = pd.merge(data, tags, left_on ='book_id', right_on = 'goodreads_book_id',how = 'inner')

for i in merged_books.columns:
    merged_books[i] = merged_books[i].fillna(' ')
for i in data.columns:
    data[i] = data[i].fillna(' ')


final_books = merged_books.groupby('book_id')['tag_name'].apply(' '.join).reset_index()

data= pd.merge(data,final_books,left_on = 'book_id', right_on = 'book_id', how= 'left')

features = ['authors','title','tag_name']

for i in features:
    data[i] = data[i].fillna(' ')

def combine_features(row):
    return row['authors']+' '+row['tag_name']+' '+row['title']


data['combined_features'] = data.apply(combine_features,axis = 1)

cv = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

count_matrix = cv.fit_transform(data['combined_features'])

similarity_score = cosine_similarity(count_matrix)

book_user_likes = input("Enter the book name that you like: ")
num = int(input("No of similar books you want to display: "))

def get_index_from_title(title):
    return data[data.title == title].index.values[0]

book_index = get_index_from_title(book_user_likes)

similar_books = list(enumerate(similarity_score[book_index]))

sorted_similar_books =  sorted(similar_books, key = lambda x:x[1], reverse = True)

def get_title_from_index(index):
    return data[data.index == index].title.values[0]

i=0
for book in sorted_similar_books:
    print(get_title_from_index(book[0]))
    i=i+1
    if i>num:
        break


