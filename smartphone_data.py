"""
Smartphone-Based Recognition of Human Activities and
 Postural Transitions Data Set 

http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions#
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from ggplot import *

base_folder = 'C:\Users\markp\OneDrive\Documents\Data Science\Projects\smartphone_movement_python\UCI HAR Dataset\\'

a_labels = pd.read_csv(
                        base_folder + 'activity_labels.txt',
                        header=None, 
                        delim_whitespace=True,
                        names=('ID','Activity'))

features = pd.read_csv(
                        base_folder + 'features.txt',
                        header=None, 
                        delimiter=r"\s+",
                        names=('ID','Sensor')
                        )

# train
trainSub = pd.read_csv(
                        base_folder + 'train\subject_train.txt',
                        header=None, 
                        names=['SubjectID'])

trainX = pd.read_csv(
                     base_folder + 'train/X_train.txt', 
                     sep='\s+', 
                     header=None)

trainY = pd.read_csv(
                    base_folder + 'train/y_train.txt',
                    sep=' ',
                    header=None)
trainY.columns = ['ActivityID']

# test
testSub = pd.read_csv(
                        base_folder + 'test\subject_test.txt',
                        header=None, 
                        names=['SubjectID'])

testX = pd.read_csv(
                     base_folder + 'test/X_test.txt', 
                     sep='\s+', 
                     header=None)

testY = pd.read_csv(
                    base_folder + 'test/y_test.txt',
                    sep=' ',
                    header=None)
testY.columns = ['ActivityID']

# combine
allSub = pd.concat([trainSub, testSub], ignore_index=True)
allX   = pd.concat([trainX, testX], ignore_index=True)
allY = trainY.append(testY, ignore_index=True) # alt method

# change column namnes
sensor_names = features.Sensor
allX.columns = sensor_names

# change activity ID for activity
allY = pd.merge(allY, a_labels, how='left', 
                left_on=['ActivityID'], right_on=['ID'])

allY = allY[['Activity']]

all = pd.concat([allSub, allX, allY], axis=1)

# save as csv
all.to_csv(base_folder + "FullData_tidy.csv")

# Now lets look at what we've got
# grouped = all.groupby (['SubjectID', 'Activity'])
# grouped = all.groupby (['Activity']).aggregate(np.std)
# grouped = all.groupby (['Activity']).describe()
# 
# ggplot(grouped, aes(x='Activity', y='std')) +\
#     geom_boxplot()

# split into train/test
xtrain, xtest, ytrain, ytest = train_test_split(all.drop('Activity', axis=1),
                                                all[['Activity']],
                                                train_size = 0.7,
                                                random_state=42
                                                )

# random forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_jobs=3)

rfc.fit(xtrain, ytrain.Activity)

predictions = rfc.predict(xtest)

confusion_matrix(ytest, predictions)

df_confusion = pd.crosstab(ytest, predictions, 
                            rownames=['Actual'],
                            colnames=['Predicted'])

df_confusion

                            
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_conf_norm

# # ROC Curve
# fpr, tpr, _ = roc_curve(ytest, predictions)
# 
# df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
# ggplot(df, aes(x='fpr', y='tpr')) +\
#     geom_line(aes(colour = 'red')) +\
#     geom_abline(aes(colour = 'blue'), 
#                 slope = 1, intercept = 0, linetype='dashed')
# # Measure AUC
# auc = auc(fpr,tpr)
# ggplot(df, aes(x='fpr', y='tpr', ymin=0, ymax='tpr')) +\
#     geom_area(alpha=0.2) +\
#     geom_line(aes(y='tpr')) +\
#     geom_abline(aes(colour = 'blue'), 
#                 slope = 1, intercept = 0, linetype='dashed') +\
#     ggtitle("ROC Curve w/ AUC=%s" % str(auc))
#     
# 
# 
# 
# 
# # use k-means to cluster
# from sklearn.cluster import KMeans
# k=6
# km = KMeans(n_clusters = k)
# km.fit(customer_deal_matrix)
