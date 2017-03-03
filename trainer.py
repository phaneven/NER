import sys
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer


# from sklearn.datasets import load_iris     # import datasets

# 0: token |1: tag |2: title or not

class trainer:
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


if __name__ == '__main__':
    trainingDataPath = sys.argv[1]
    classifierPath = sys.argv[2]
    with open(trainingDataPath, 'rb') as f:
        training_set = pickle.load(f)
    # print (training_set)
    # find feature
    tr = trainer(training_set);
    tr.collectFeature();

    vectorizer = DictVectorizer()
    X = tr.features_and_tags
    # print(X)
    y = tr.classList
    X = vectorizer.fit_transform(X)

    # training data
    classifier = LogisticRegression(C=3,penalty='l1')
    trained = classifier.fit(X, y)
    print(classifier.score(X, y))
    print(classifier.predict(X))

    with open(classifierPath, 'wb') as f:
        pickle.dump(trained, f)

    with open('./vectorizer_data', 'wb') as f:
        pickle.dump(vectorizer,f)

    # with open(classifierPath, 'rb') as f:
    #     a = pickle.load(f)
    # print(a)


