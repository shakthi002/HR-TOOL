import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from similarity import calc_similarity
from sklearn.linear_model import LogisticRegression


def create_df(resumes,jd):
    '''
    Function to create dataframe for given resumes and job description
    :param resumes:
    :return Pandas Dataframe:
    '''
    cols = list(set(jd))
    temp = {}
    for resume in resumes:
        for token in cols:
            if token in resume:
                if token not in temp:
                    temp[token] = [1]
                else:
                    temp[token].append(1)
            else:
                if token not in temp:
                    temp[token] = [0]
                else:
                    temp[token].append(0)
    df = pd.DataFrame(temp)
    return df

def clustering(jd,df):
    '''
    Function to cluster the dataset into two clusters using K means clustering
    :param df:
    :return:
    '''
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
    labels = np.array(kmeans.labels_)
    idx = np.argwhere(labels==1)
    max_sm_c1 = 0
    df["Label"] = np.zeros(len(labels))
    for index in idx:
        max_sm_c1 = max(max_sm_c1,calc_similarity([1]*df.shape[1],list(df.iloc[index])))
    max_sm_c2 = 0
    for index in range(len(labels)):
        if index not in idx:
            max_sm_c2 = max(max_sm_c2, calc_similarity([1]*df.shape[1], list(df.iloc[index])))

    if max_sm_c1>max_sm_c2:
        #cluster with label 1 will have label as 1->matching resume
        for index in idx:
            df["Label"].iloc[index] = 1
    else:
        # cluster with label 0 will have label as 1->matching resume
        for index in range(len(labels)):
            if index not in idx:
                df["Label"].iloc[index] = 1

    return df


def logistic_reg(jd,df):
    '''
    Function to fit logistic regression to dataset to get the prob
    :param df:
    :return: ranked dataframe
    '''

    X = df.drop(columns=["Label"])
    Y = df["Label"]
    df["Rank"] = np.zeros(len(Y))
    clf = LogisticRegression()
    clf.fit(X,Y)

    prob = {}
    for idx in range(len(X)):
        prob[idx] = (clf.predict_proba((np.array(X.iloc[idx])).reshape(1,-1)))[0][1]
        print(prob)

    prob = dict(sorted(prob.items(), key=lambda item: item[1],reverse=False))

    cnt = 1
    for idx in prob:
        df["Rank"].iloc[idx] = cnt
        cnt+=1

    return df

def main_model(jd,resumes):
    df = create_df(resumes,jd)
    df = clustering(jd,df)
    df = logistic_reg(jd,df)

    return df






