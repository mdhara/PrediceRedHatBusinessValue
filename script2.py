import pandas as pd
import datetime
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def processPeople(people):
	people['people_id'] = people['people_id'].apply(lambda x: (x.split('_')[1]))
	people['people_id'] = pd.to_numeric(people['people_id']).astype(int)
	
	people['year'] = people['date'].dt.year
	people['month'] = people['date'].dt.month
	people['day'] = people['date'].dt.day
	people.drop('date', axis=1, inplace=True)

	cols = list(people.columns)
	bools = cols[11:39]
	strs = cols[1:11]
	for col in bools:
		people[col] = pd.to_numeric(people[col]).astype(int)

	for col in strs:
		people[col] = people[col].fillna('type 0')
		people[col] = people[col].apply(lambda x: x.split(' ')[1])
		people[col] = pd.to_numeric(people[col]).astype(int)

	return people


def preprocessAct(train, train_set=True):
	train['year'] = train['date'].dt.year
	train['month'] = train['date'].dt.month
	train['day'] = train['date'].dt.day
	train = train.drop(['activity_id', 'date'], axis=1)
	if(train_set):
		train = train.drop(['outcome'], axis=1)
	train['people_id'] = train['people_id'].apply(lambda x: (x.split('_')[1]))
	train['people_id'] = pd.to_numeric(train['people_id']).astype(int)

	cols = list(train.columns)
	strs = cols[1:12]
	for col in strs:
		train[col] = train[col].fillna('type 0')
		train[col] = train[col].apply(lambda x: x.split(' ')[1])
		train[col] = pd.to_numeric(train[col]).astype(int)

	return train


def training(train, labels, test):
	test_percent = 0.2
	X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=test_percent, random_state=23)
	n_features = len(test.columns)-1
	##train
	model= RandomForestClassifier(n_estimators=100,max_features=int(round(math.sqrt(n_features))))
	model.fit(X_train, y_train)

	##training predictions
	proba = model.predict_proba(X_test)
	preds = proba[:,1]
	score = roc_auc_score(y_test, preds)
	print("Area under ROC {0}".format(score))

	##test predictions
	test_proba = model.predict_proba(test)
	test_preds = test_proba[:,1]

	return test_preds

def intersect(a, b):
    return list(set(a) & set(b))

def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    return pd.DataFrame(output)

def modelfit(train, labels, test, features, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	param_test1 = {
 		'max_depth':range(3,10,2),
 		'min_child_weight':range(1,6,2)
	}
	model = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


	test_percent = 0.2
	X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=test_percent, random_state=23)

	xgb_param = model.get_xgb_params()


	#Fit the algorithm on the data
	model.fit(X_train, y_train)
	print(model.grid_scores_)
	print(model.best_params_)
	print(model.best_score_)
	##training predictions
	proba = model.predict_proba(X_test)
	preds = proba[:,1]
	score = roc_auc_score(y_test, preds)
	print("Area under ROC {0}".format(score))

    #Print model report:
#	print "\nModel Report"
#	print "Accuracy : %.4g" % accuracy_score(y_train, preds)
#	print "AUC Score (Train): %f" % roc_auc_score(y_train, preds)
                    
	feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
#	plt.show()

	##test predictions
	test_proba = model.predict_proba(test)
	test_preds = test_proba[:,1]

	return test_preds



def submit(id,outcome):
	output = pd.DataFrame({ 'activity_id' : id, 'outcome': outcome })
	output.to_csv("submission.csv", index=False)

def main():
	print("Reading act_train")
	train  = pd.read_csv('act_train.csv', parse_dates=['date'])
	labels = train['outcome']
	print("preprocessing act_train")
	train  = preprocessAct(train)

	print("Reading act_test")
	test   = pd.read_csv('act_test.csv', parse_dates=['date'])
	test_ids = test['activity_id']
	print("preprocessing act_test")
	test   = preprocessAct(test, train_set=False)

	print("Reading people")
	people = pd.read_csv('people.csv', parse_dates=['date'])
	print("preprocessing people")
	people = processPeople(people)

	print("Merging training data")
	train  = train.merge(people, how='left', on='people_id')
	print("Merging test data")
	test   = test.merge(people, how='left', on='people_id')

	train = train.drop(['people_id'], axis=1)
	test  = test.drop(['people_id'], axis=1)

	features = [x for x in train.columns]
#	features = get_features(train, test)

	print("training model")
	outcome = modelfit(train,labels,test,features)

	print("Creating Submission")
	submit(test_ids,outcome)


if __name__ == "__main__":
    main()
