import pickle 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

train_copy_df = pd.read_pickle('train_cleaned_df.pk')
test_copy_df = pd.read_pickle('test_cleaned_df.pk')

remove_cols = 	['Name',
				'PetID',
                'RescuerID',
                #'Type',
                'NameFrequency']

X = train_copy_df.drop(remove_cols, axis=1)

X = X.drop(['AdoptionSpeed'],axis=1)

y = train_copy_df['AdoptionSpeed']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


clf1 = RandomForestClassifier(n_estimators=500,max_depth=20,random_state=0)
clf1.fit(x_train,y_train)
y_pred = clf1.predict(x_test)
print('Test Accuracy:',accuracy_score(y_test,y_pred))

y_train_pred = clf1.predict(x_train)
print('Train Accuracy: ',accuracy_score(y_train,y_train_pred))


TEST = test_copy_df.drop(remove_cols, axis=1)

y_final_pred = clf1.predict(TEST)

petid = test_copy_df['PetID']
final_pred = pd.Series(y_final_pred,name='AdoptionSpeed')
submission = pd.concat([petid,final_pred],axis=1)

submission.to_csv('submission.csv',index=False)