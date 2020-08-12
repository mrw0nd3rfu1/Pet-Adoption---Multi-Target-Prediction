  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
import numpy as np

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['bayes'] = GaussianNB()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

X = pd.read_csv('Train.csv')
X_test = pd.read_csv('Test.csv')
X_sample = pd.read_csv('Test.csv')

X.fillna(X.mean(),inplace=True)
Y= X.breed_category
X.drop(['breed_category','pet_category','issue_date','listing_date','pet_id'],axis=1,inplace=True)
###
X_test.drop(['issue_date','listing_date','pet_id'],axis=1,inplace=True)

Col_to_encode = ['color_type']

X[Col_to_encode]= X[Col_to_encode].apply(lambda col:LabelEncoder().fit_transform(col))
###
X_test[Col_to_encode]= X_test[Col_to_encode].apply(lambda col:LabelEncoder().fit_transform(col))
X_test.fillna(X_test.mean(), inplace=True)
#train_X,val_X,train_Y,val_Y = train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=0)

#after checking all the models best model is used
model_random = DecisionTreeClassifier()

#checking the model accuracy and mae scores
# models = get_models()
# results,names = list(),list()
# for name, model in models.items():
# 	scores = evaluate_model(model, X, Y)
# 	results.append(scores)
# 	names.append(name)
# 	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

model_random.fit(X,Y)
preds = model_random.predict(X_test)
#print(preds.astype(int))
output = pd.DataFrame({'pet_id':X_sample.pet_id,'condition':X_test.condition,'color_type':X_test.color_type,'length(m)':X_test.length,'height(cm)':X_test.height,'X1':X_test.X1,'X2':X_test.X2,'breed_category':preds.astype(int)})
output.to_csv('new_test.csv', index=False)
# me = mean_absolute_error(val_Y,preds)
# r_score = r2_score(val_Y,preds)
# print(me,r_score)