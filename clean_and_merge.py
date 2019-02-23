import numpy as np
import pandas as pd 
import pickle

data_path = "/Users/raghavatal/Developer/Data/Petfinder_DataMiningProject/petfinder-adoption-prediction"


train_df = pd.read_csv(data_path+'/train.csv')
test_df = pd.read_csv(data_path + "/test/test.csv")


train_copy_df = train_df.copy()
test_copy_df = test_df.copy()


def countNanInColumn(df,col):
    count = df.loc[df[col].isnull(),col].shape[0]
    return count

def replaceNanWithValue(df,col,value):
    count = countNanInColumn(df,col)
    df.loc[df[col].isnull(),col] = value
    return count


count = replaceNanWithValue(train_copy_df,'Name','No Name')
#print('Values Replaced: {}'.format(count))

count = replaceNanWithValue(test_copy_df,'Name','No Name')
#print('Values Replaced: {}'.format(count))


train_copy_df['NumColors'] = train_copy_df.loc[:,'Color1':'Color3'].apply(lambda row: bool(row.Color1)
                                                         + bool(row.Color2) 
                                                         + bool(row.Color3),axis=1)

test_copy_df['NumColors'] = test_copy_df.loc[:,'Color1':'Color3'].apply(lambda row: bool(row.Color1)
                                                         + bool(row.Color2) 
                                                         + bool(row.Color3),axis=1)

train_copy_df['IsMixedBreed'] = train_copy_df.loc[:,'Breed1':'Breed2'].apply(lambda row: bool(row.Breed1)*bool(row.Breed2),axis=1)
test_copy_df['IsMixedBreed'] = test_copy_df.loc[:,'Breed1':'Breed2'].apply(lambda row: bool(row.Breed1)*bool(row.Breed2),axis=1)


nameFrequency_df = pd.DataFrame(train_copy_df['Name'].value_counts())
train_copy_df['NameFrequency'] = nameFrequency_df.loc[train_copy_df['Name'],'Name'].values


nameFrequency_df = pd.DataFrame(test_copy_df['Name'].value_counts())
test_copy_df['NameFrequency'] = nameFrequency_df.loc[test_copy_df['Name'],'Name'].values

sentiment_df = pd.read_csv(data_path+'/Processed/sentiment.csv')
sentiment_df.set_index('PetID')
(left ,right, key)  = (train_copy_df ,sentiment_df[['PetID','Magnitude','Score']],'PetID')
train_copy_df=pd.merge(left, right, on=key,how='left')


(left ,right, key)  = (test_copy_df ,sentiment_df[['PetID','Magnitude','Score']],'PetID')
test_copy_df=pd.merge(left, right, on=key,how='left')


train_copy_df.loc[train_copy_df['Magnitude'].isnull(),'Magnitude'] = 0.0
train_copy_df.loc[train_copy_df['Score'].isnull(),'Score'] = 0.0

test_copy_df.loc[test_copy_df['Magnitude'].isnull(),'Magnitude'] = 0.0
test_copy_df.loc[test_copy_df['Score'].isnull(),'Score'] = 0.0


train_copy_df.loc[train_copy_df['Description'].isnull(),'Description'] = ''
test_copy_df.loc[test_copy_df['Description'].isnull(),'Description'] = ''

train_copy_df['WordCount'] = train_copy_df['Description'].str.split().apply(lambda x: len(x))
test_copy_df['WordCount'] = test_copy_df['Description'].str.split().apply(lambda x: len(x))


train_copy_df = train_copy_df.drop('Description',axis=1)
test_copy_df = test_copy_df.drop('Description',axis=1)


train_copy_df['AdoptionSpeed'] = train_copy_df['AdoptionSpeed'].astype('int',errors='ignore').fillna(0)



train_copy_df.to_pickle('train_cleaned_df.pk')
test_copy_df.to_pickle('test_cleaned_df.pk')

