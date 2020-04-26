from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


#loading dataset and label encoding target
from MajorityVoteClassifier import MajorityVoteClassifier

iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state= 8)

# here we use logistic regression, decisoion tree classifier, knn

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import  Pipeline
import numpy as np

clf1 = LogisticRegression(penalty='l2', C=0.01, random_state= 1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3 = Pipeline([['sc',StandardScaler()],['clf',clf3]])

clf_labels = [
    'Logistic Regression',
    'Decision Tree',
    'KNN'
]

print('10 fold cross validation \n')

for clf,label in zip([pipe1,clf2,pipe3], clf_labels):
    scores = cross_val_score(estimator = clf, X = X_train, y = y_train, cv = 10, scoring= 'roc_auc')
    print("ROC AUC : %0.2f ( +/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

print("After adding Majority Voting Classifier")
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
clf_labels +=['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,X = X_train, y = y_train, cv =10, scoring='roc_auc')
    print("ROC AUC : %0.2f ( +/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))