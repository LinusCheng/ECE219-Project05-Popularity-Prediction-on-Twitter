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



##############################################################################
##############################################################################
                     """ Classifiers """
##############################################################################
##############################################################################
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import confusion_matrix,roc_curve,auc

from sklearn import svm,linear_model,naive_bayes



#""" SVM """
model_svm = svm.SVC(kernel='linear').fit(X_SVD_train,Y_train)
Y_pred = model_svm.predict(X_SVD_test)

print("Accuracy =", np.mean(Y_pred == Y_test))
#print("Accuracy =", np.mean(Y_pred == np.asarray(Y_test))) # List & np.array can element wise match
#Accuracy = 0.904449565679844
# No hyperparameter, so no cross validation


print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))

#             precision    recall  f1-score   support
#
#         WA       0.92      0.95      0.93      3992
#         MA       0.87      0.79      0.83      1649
#
#avg / total       0.90      0.90      0.90      5641


print(confusion_matrix(Y_test,Y_pred))

#[[3800  192]
# [ 347 1302]]



test_score=model_svm.decision_function(X_SVD_test)

fpr, tpr, _ = roc_curve(Y_test, test_score)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("SVM ROC curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")



################################################################################################
""" SVM Penalty parameter = 20 """




model  = svm.SVC(C=20,kernel='linear').fit(X_SVD_train,Y_train)
Y_pred = model.predict(X_SVD_test)


print("Accuracy =", np.mean(Y_pred == Y_test))
print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))
print("confusion matrix")
print(confusion_matrix(Y_test,Y_pred))


#Accuracy = 0.9035631980145364
#             precision    recall  f1-score   support
#
#         WA       0.91      0.95      0.93      3992
#         MA       0.87      0.78      0.83      1649
#
#avg / total       0.90      0.90      0.90      5641
#
#[[3805  187]
# [ 357 1292]]



test_score=model.decision_function(X_SVD_test)

fpr, tpr, _ = roc_curve(Y_test, test_score)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("SVM Hard ROC curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")






################################################################################################
#""" SVM Penalty parameter = 50 """




model  = svm.SVC(C=50,kernel='linear').fit(X_SVD_train,Y_train)
Y_pred = model.predict(X_SVD_test)


print("Accuracy =", np.mean(Y_pred == Y_test))
print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))
print("confusion matrix")
print(confusion_matrix(Y_test,Y_pred))

#Accuracy = 0.904449565679844
#             precision    recall  f1-score   support
#
#         WA       0.92      0.95      0.93      3992
#         MA       0.87      0.79      0.83      1649
#
#avg / total       0.90      0.90      0.90      5641
#
#confusion matrix
#[[3801  191]
# [ 348 1301]]




test_score=model.decision_function(X_SVD_test)

fpr, tpr, _ = roc_curve(Y_test, test_score)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("SVM Hard ROC C=50 curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")


################################################################################################
""" SVM Penalty parameter = 100 """



model  = svm.SVC(C=100,kernel='linear').fit(X_SVD_train,Y_train)
Y_pred = model.predict(X_SVD_test)


print("Accuracy =", np.mean(Y_pred == Y_test))
print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))
print("confusion matrix")
print(confusion_matrix(Y_test,Y_pred))

#Accuracy = 0.9046268392129055
#             precision    recall  f1-score   support
#
#         WA       0.92      0.95      0.93      3992
#         MA       0.87      0.79      0.83      1649
#
#avg / total       0.90      0.90      0.90      5641
#
#
#confusion matrix
#[[3801  191]
# [ 347 1302]]


test_score=model.decision_function(X_SVD_test)

fpr, tpr, _ = roc_curve(Y_test, test_score)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("SVM Hard ROC C=100 curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")




################################################################################################
""" Logistic """



model  = linear_model.LogisticRegression().fit(X_SVD_train,Y_train)
Y_pred = model.predict(X_SVD_test)


print("Accuracy =", metrics.accuracy_score(Y_pred , Y_test))
print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))
print("confusion matrix")
print(confusion_matrix(Y_test,Y_pred))


#Accuracy = 0.9053359333451516
#             precision    recall  f1-score   support
#
#         WA       0.92      0.95      0.93      3992
#         MA       0.87      0.79      0.83      1649
#
#avg / total       0.90      0.91      0.90      5641
#
#confusion matrix
#[[3798  194]
# [ 340 1309]]



test_score=model.decision_function(X_SVD_test)

fpr, tpr, _ = roc_curve(Y_test, test_score)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("Logistic ROC curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")



#############################################################################################
""" Naive Bayes """

model = naive_bayes.GaussianNB().fit(X_SVD_train,Y_train)


Y_pred = model.predict(X_SVD_test)


print("Accuracy =", metrics.accuracy_score(Y_pred , Y_test))
print(metrics.classification_report(Y_test, Y_pred, target_names= ['WA','MA']))
print("confusion matrix")
print(confusion_matrix(Y_test,Y_pred))


#Accuracy = 0.8431129232405602
#             precision    recall  f1-score   support
#
#         WA       0.86      0.93      0.89      3992
#         MA       0.78      0.64      0.71      1649
#
#avg / total       0.84      0.84      0.84      5641
#
#confusion matrix
#[[3698  294]
# [ 591 1058]]


test_prob = model.predict_proba(X_SVD_test)[:, 1]

fpr, tpr, _ = roc_curve(Y_test, test_prob)
fig, ax = plt.subplots()
roc_auc = auc(fpr,tpr)
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
ax.grid(color='0.7', linestyle='--', linewidth=1)
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower right")
ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
plt.title("Naive Bayes ROC curve")
plt.xlabel("FP rate")
plt.ylabel("TP rate")



print("completed")




