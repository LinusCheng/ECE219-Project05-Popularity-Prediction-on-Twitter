import numpy as np
#import matplotlib.pyplot as plt
import os
os.chdir('E:/219_data')


import json
#import nltk
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD






"""  ======  """
#twt_sb = open('train/tweets_#superbowl.txt', encoding="utf8")
#read first line
#print(twt_sb.readline())
# number of lines
#print(sum(1 for line in twt_sb))
#1348766+1 lines
#twt_sb.close()


              
twt_sb = open('train/tweets_#superbowl.txt', encoding="utf8")                            
all_lines = twt_sb.readlines()
print("all_lines got!")
twt_sb.close()
                     


                     
WA = ['washington'   , 'wash','wa','seattle','kirkland' ]
MA = ['massachusetts', 'mass','ma','boston' ,'cambridge']

#break strings
token = RegexpTokenizer(r'\w+')




def get_location(all_lines):
    twt_list      = []
    location_list = []
    for i, line in enumerate(all_lines):
        twt_i      = json.loads(line)
        location_i = twt_i['tweet']['user']['location']
        loc = token.tokenize(location_i.lower())
        if  (any(x in loc for x in WA)):
            twt_list.append(twt_i['title'])
            location_list.append(0)
        elif (any(x in loc for x in MA)):
            twt_list.append(twt_i['title'])
            location_list.append(1)
        else:
            pass
#        if i %10000 ==0:
#            print("tweet:",i)
    print("twt_list & location_list generated!")
    print("WA:0 MA:1")
    return twt_list,location_list

twt_list,Y = get_location(all_lines)

del all_lines


# 56410 of the 1348767 lines have WA or MA

""" Lemmatisation """

#WordNetLemmatizer().lemmatize("This is a book books an cat cats")

def lemm_tweeks(twt_list):
    twt_lemm=[]
    for twt_i in twt_list:
        words = token.tokenize(twt_i)   
        words_lem_i = []
        for words_j in words:
            words_lem_j = WordNetLemmatizer().lemmatize(words_j)
            words_lem_i.append(WordNetLemmatizer().lemmatize(words_lem_j,'v'))
        twt_lemm.append(' '.join(w for w in words_lem_i))
    return twt_lemm
twt_lemm = lemm_tweeks(twt_list)


""" feature_extraction split train and test data"""
twt_len_10div = int(np.floor(len(twt_list)/10))
vectorizer = TfidfVectorizer(min_df=1,stop_words=text.ENGLISH_STOP_WORDS)

X_tfidf_train = vectorizer.fit_transform(twt_lemm[twt_len_10div+1:]  )
X_tfidf_test  = vectorizer.transform(    twt_lemm[0:twt_len_10div]  )

X_tfidf       =  vectorizer.fit_transform(twt_lemm)

Y_train = Y[twt_len_10div+1:]
Y_test  = Y[0:twt_len_10div]


print('TFIDF done!')


""" SVD """
svd = TruncatedSVD(n_components=50)
X_SVD_train = svd.fit_transform(X_tfidf_train)
X_SVD_test  = svd.transform(X_tfidf_test)

X_SVD       = svd.fit_transform(X_tfidf)

print('SVD done!')

del twt_len_10div,twt_lemm



Y       = np.asarray(Y)
Y_train = np.asarray(Y_train)
Y_test  = np.asarray(Y_test)








