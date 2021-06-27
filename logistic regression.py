# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:41:21 2019

@author: amris
"""

import pandas as pn

import numpy as nm
import seaborn as sn
dp =pn.read_csv(r'C:\Users\amris\Downloads\diabetes.csv')
dp.columns
y= pn.DataFrame(dp.Outcome)
x=dp.drop('Outcome',axis=1)
from sklearn.linear_model import LogisticRegression
obj=LogisticRegression()
obj.fit(x,y)
pr=obj.predict(x)
from sklearn.metrics import confusion_matrix
a=confusion_matrix(y,pr)
acc=(a[0,0]+a[1,1])/768
print(acc*100)
 
a.shape
