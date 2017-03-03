import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt



n_folds = 10
#fetch feature
class featureFetcher:
    features_and_tags = []
    trainingData = []
    classList = []

    def __init__(self, training_set):
        self.features_and_tags = []
        self.trainingData = training_set

    ################# feature #################
    def checkTitle (self, token):
        if (token[2] == 'TITLE'):
            return 1
        return 0


    # feature: abbreviation, e.g. CEO; Mr.
    def checkAbbreviation (self, token):
        if len(token[0]) < 2:  # e.g. 'I' is a special one
            return 0
        elif token[0][0].isupper() and token[0][-1] == '.':
            if len(token[0]) > 2:
                return 1
        # elif token[0][0].isupper() and token[0][-1].isupper():
        #     return 1
        return 0

    # feature: tag, if it is n. it has some possibility to be a title
    def checkTag (self, token):
        tag = token[1]
        if tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS":
            return 1
        else:
            return 0

    # feature: affix e.g. director, minerster
    def checkAffix(self, token):
        affix_list = ['er', 'or', 'ist', 'an', 'ive', 'ant']
        wnl = WordNetLemmatizer()
        word = wnl.lemmatize(token[0])
        if word[-2:] in affix_list or word[-3:] in affix_list:
            if self.checkTag(token) == 1:
                return 1
            else:
                return 0
        else:
            return 0

    # feature: previous word
    def checkPreviousWord (self, sentence, index) :
        if index == 0:
            return 0
        else:
            if self.checkTag(sentence[index-1]) == 1:
                return 1
            else:
                return 0

    def checkNextWord(self, sentence, index):
        if index == len(sentence) - 1:
            return 0
        else:
            if self.checkTag(sentence[index + 1]) == 1:
                return 1
            else:
                return 0

    # start to build feature_and_tags list
    def collectFeature (self):
        for sentence in self.trainingData:
            index = 0
            for token in sentence:
                feature_dict = {}  # used to store every word's feature
                ####### feature ########
                feature_dict[token[0]] = 1
                feature_dict['previous_tag'] = self.checkPreviousWord(sentence,index)
                feature_dict['next_tag'] = self.checkNextWord(sentence, index)
                feature_dict['abbr'] = self.checkAbbreviation(token)
                feature_dict['Affix'] = self.checkAffix(token)
                ########################
                self.features_and_tags.append(feature_dict)
                self.classList.append(self.checkTitle(token))
                index += 1


    ###########################################

def test_LogisticRegression(Cvalue, train_X, train_y, test_X, test_y):
    lr = LogisticRegression(C=Cvalue, penalty='l1')
    lr.fit(train_X, train_y)
    # train_error = f1_score(train_y, lr.predict(train_X))
    # test_error = f1_score(test_y, lr.predict(test_X))
    train_error = lr.score(train_X, train_y)
    test_error = lr.score(test_X, test_y)
    return train_error, test_error

def cv_LogisticRegression(Cvalue):
    train_total_error = 0
    test_total_error = 0

    for train, test in kf:
        train_X = feature_list[train]
        train_Y = class_list[train]
        test_X = feature_list[test]
        test_Y = class_list[test]
        train_error, test_error = test_LogisticRegression(Cvalue, train_X, train_Y, test_X, test_Y)
        train_total_error += train_error
        test_total_error += test_error

    return train_total_error / n_folds, test_total_error / n_folds

if __name__ == '__main__':
    training_data_path = "training_data"

    with open(training_data_path, "rb") as f:
        data_set = pickle.load(f)

    FF = featureFetcher(data_set)
    FF.collectFeature()
    feature_list = FF.features_and_tags
    class_list = FF.classList
    class_list = pd.Series(class_list)
    vectorizer = DictVectorizer()
    feature_list = vectorizer.fit_transform(feature_list)
    kf = KFold(n=feature_list.shape[0], n_folds=n_folds, shuffle=True, random_state=42)

    rng = [0.1,0.2,0.3,0.4,0.5]
    rng += sorted([0.5] + list(range(1, 20)))
    cv_res = []
    for i in rng:
        train_error, test_error = cv_LogisticRegression(i)
        cv_res.append([i, train_error, test_error])

    print(cv_res)
    cv_res_arr = np.array(cv_res)
    plt.figure(figsize=(16, 9))
    plt.title('Error vs. Cvalue')
    plot_train, = plt.plot(cv_res_arr[:, 0], cv_res_arr[:, 1], label='training')
    plot_test, = plt.plot(cv_res_arr[:, 0], cv_res_arr[:, 2], label='testing')
    plt.legend(handles=[plot_train, plot_test])
    plt.ylim((min(min(cv_res_arr[:, 1]), min(cv_res_arr[:, 2])) - 0.01,
              max(max(cv_res_arr[:, 1]), max(cv_res_arr[:, 2])) + 0.01))
    plt.xticks(rng)
    plt.show()
