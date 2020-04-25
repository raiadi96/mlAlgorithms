import numpy as np
from sklearn.base import BaseEstimator,clone  #adds basic getter and setter functions
from sklearn.base import ClassifierMixin 
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six #adds support for usage in both python2 and 3import
from sklearn.pipeline import  _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, classifiers, vote = 'classLabel', weights = None):

        self.clasifiers = classifiers
        self.named_classifier = {key:value for key,value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights


    def fit(self,X,y):

       self.lablelenc_ = LabelEncoder()
       self.lablelenc_.fit(y)
       self.clasifiers = self.lablelenc_.classes_
       self.clasifiers_ = []
       for clf in self.clasifiers:
           fitted_clf = clone(clf).fit(X,self.labelenc_.transform(y))
           self.clasifiers_.append(fitted_clf)
       return self

    def predict(self,X):
        if self.vote == 'probablity':
            maj_vote = np.argmax(self.predict_proba(X))
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.clasifiers_]).T
            maj_vote = np.apply_along_axis(lambda X: np.bincount(X,weights=self.weights)) #look into this later
            maj_vote = self.lablelenc_.inverse_transform(maj_vote)
            return maj_vote

    def predict_proba(self, X):
        probas = np.asarray([clf.predict(X) for clf in self.clasifiers_])
        avg_probas = np.average(probas, axis = 0, weights= self.weights)
        return avg_probas

    def get_params(self, deep=True):
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifier.copy()
            for name,step in six.iteritems(self.named_classifiers):
                for key,value in six.iteritems(step.get_params(deep=True)):
                    out["%s___%s" % (name,key)] = value
            return out


