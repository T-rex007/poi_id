#!/usr/bin/python
'''

Tyrel S.I Cadogan
Udacity final project
person of interest identifier
'''
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [ 'poi','restricted_stock', 'restricted_stock_deferred','shared_receipt_with_poi','total_stock_salary_ratio','bonus','salary','total_stock_value','long_term_incentive','msg_ratio_from','msg_ratio_to',] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#########################################################################################################################
### Task 2: Remove outliers

cleaned_data = data_dict
del cleaned_data['TOTAL']
cleaned_data['LAVORATO JOHN J']['bonus'] = 0

#########################################################################################################################

### Task 3: Create new feature(s)

from enr_tools import Ratio
for person in cleaned_data:

	features = cleaned_data[person]
	features['total_stock_salary_ratio'] = Ratio(features['total_stock_value'],features['salary'])
	features['msg_ratio_from'] = Ratio(features['from_this_person_to_poi'], features['from_messages'])
	features['msg_ratio_to'] = Ratio(features['from_poi_to_this_person'], features['to_messages'])

#########################################################################################################################
### Elucidating dataset

import pandas as pd

print '-Number of people in dataset', len(cleaned_data)
print '-Number of features used', len(features_list)

df = pd.DataFrame(cleaned_data)
poi = df.loc['poi']
poi = poi.as_matrix()
print poi
nofp = 0
for x in poi:
	if x == True:
		nofp = nofp + 1
	else:
		pass

print '-The number of poi in dataset is %d, %f percent of the dataset.'%(nofp, (float(nofp)/float(len(cleaned_data))*100))
print '-The number of non poi in the dataset is', (len(cleaned_data) - nofp)

#########################################################################################################################
### Store to my_dataset for easy export below.			 
my_dataset = cleaned_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### feature selection

from sklearn.feature_selection import SelectKBest
feat_sel = SelectKBest(k=6)
features = feat_sel.fit_transform(features, labels)

print 'Feature\'s pvalues: ', feat_sel.pvalues_
print 'Feature\'s scores: ', feat_sel.scores_

########################################################################################################################
### Feature scaling

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
features = std_scaler.fit_transform(features)


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size = 0.4, random_state = 13)

########################################################################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import  confusion_matrix, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


print
#######################################################################################################
### GaussianNB
print 'GaussianNB'
clfg = GaussianNB()

clfg.fit( features_train, labels_train)
pred = clfg.predict(features_test)
acc = accuracy_score( labels_test, pred)
prec= precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)
print
#######################################################################################################
### Logistic Regression
print 'LogisticRegression'

clfl = LogisticRegression(penalty = 'l1', C = 2, class_weight = {0:0.1, 1:0.3})

clfl.fit(features_train,labels_train)
pred = clfl.predict(features_test)
acc = accuracy_score( labels_test, pred)
prec= precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)
print
#######################################################################################################
print 'BaggingClassifier'

clfs = BaggingClassifier(DecisionTreeClassifier(max_depth =2, class_weight ={ 0:0.1, 1: 0.3}), n_estimators =50, bootstrap = True,oob_score = True)
clfs.fit (features_train, labels_train)
pred = clfs.predict(features_test)
acc = accuracy_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)
#######################################################################################################
###DecisionTreeClassifier
print 'RandomForestClassifier'

clfD = RandomForestClassifier(max_depth = 2, random_state = 2,class_weight ={0:0.1, 1:0.6})

clfD.fit(features_train,labels_train)
pred = clfD.predict(features_test)
acc = accuracy_score( labels_test, pred)
prec= precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)
#######################################################################################################
### SGDClassifier
print 'SGSDClassifier'

clfv = SGDClassifier(loss='modified_huber', penalty ='l2', random_state = 42 ,class_weight= {0: 0.1, 1:0.2})

clfv.fit(features_train, labels_train)
pred = clfv.predict(features_test)
acc = accuracy_score( labels_test, pred)
prec= precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)

#####################################################################################################
### Ensemble( Voting Classifier)
print 'Ensemble(VotingClassifier)'

clf = VotingClassifier(estimators = [('lg', clfl), ('nb', clfg), ('sgdc', clfv), ('dec',clfD), ('svc',clfs)], voting = 'hard')

clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
acc = accuracy_score( labels_test, pred)
prec= precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)

print 'The accuracy is;',acc
print 'The precision is:',prec
print 'the recall is ', rec
print "Confucion Matrix"
print confusion_matrix( labels_test, pred)

#######################################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


