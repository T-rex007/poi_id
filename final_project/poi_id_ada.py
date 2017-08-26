#!/usr/bin/python
import numpy as np
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [ 'poi','restricted_stock','salary','bonus','total_stock_value','shared_receipt_with_poi', 'ratio_of_sent_messages','ratio_of_recievedmsgs','inverse_expenses'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

import pandas as pd





'''
for k, v in data_dict.iteritems():
	print v
'''

from sklearn.decomposition import PCA

	
### Task 2: Remove outliers

### Task 3: Create new feature(s)
to_mess = []
from_mess = []
from_poi_to_this_person = []
from_this_person_to_poi = []
expenses = []

### Store to my_dataset for easy export below.

from decimal import *

### Get msg value and store them in a list
for person, features in data_dict.iteritems():
	for feature, value in features.iteritems():
		if feature == 'expenses':
			expenses.append(value)
		if feature == 'to_messages':
			to_mess.append(value)
		if feature == 'from_messages':
			from_mess.append(value)
		if feature == 'from_poi_to_this_person':
			from_poi_to_this_person.append(value)
		if feature == 'from_this_person_to_poi':
			from_this_person_to_poi.append(value)
		else:
			pass

for person, features in data_dict.iteritems():
	for value in expenses:
		if value == 'NaN':
			ex= 0
			features.update({'inverse_expenses':ex})	
		else:
			ex = Decimal(1)/Decimal(value)
			features.update({'inverse_expenses':ex})

	for den, num in zip( to_mess, from_poi_to_this_person ):
		if num =='NaN' or den =='NaN':
			rat = {'ratio_of_recievedmsgs':0.00}
			features.update(rat)

		else:
			rat = Decimal(num)/Decimal(den)

			features.update({'ratio_of_recievedmsgs':rat})

	for num, den in zip( from_this_person_to_poi, from_mess):
		if num == 'NaN' or den == 'NaN':
			rat = {'ratio_of_sent_messages':0.00}
			features.update(rat)

		else:

			rat = Decimal(num)/Decimal(den)
			features.update({'ratio_of_sent_messages':rat})

			
			 

df = pd.DataFrame(data_dict)
print df

my_dataset = data_dict
def do_pca():
	from sklearn.decomposition import PCA
	pca = PCA(n_components = 2)
	pca.fit(my_dataset)
	return pca

pca = do_pca()
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_data = pca.transfrom(my_dataset)
for x, a  in zip(transformed_data, data):
	plt.scatter( first_pc[0]*x(0), first_pc[1]*x[0], color = 'r')
	plt.scatter(second_pc[0]*x[1], second_pc[1]*x[1], color = 'c')
	plt.scatter(a[0], a[1], color ='b')

plt.show()
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
############################################################
pre_pipe = Pipeline([
	('imp', Imputer(missing_values =0,strategy = 'median')),
	
	(' var_thres', VarianceThreshold()),
	('std_scaler', StandardScaler() ),
	])

features = pre_pipe.fit_transform(features)

from sklearn.preprocessing import PCA
pca = PCA()


### Removing outliers #TO DO
'''
feat = pd.DataFrame(features)
#feat= feat.drop(8, 0)
print feat
p=feat 
feat = feat.as_matrix()
labe = pd.DataFrame(labels)
#labe = labe.drop(, 0)
labe = labe.as_matrix()
df1 = p.T
df2 = df1.loc[0]


'''

### Creating new features 






print features
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size = 0.4, random_state = 42)



from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# make_pipeline(StandardScaler(), PCA(), SVC.fit_tranform(), DecisionTreeClassifier())

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
### gridsearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import  confusion_matrix


clf = AdaBoostClassifier(base_estimator= LogisticRegression(),algorithm ='SAMME')
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

acc = accuracy_score( labels_test, pred)
print 'The accuracy is ',acc


print "Confucion Matrix"
print confusion_matrix( labels_test, pred)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)








