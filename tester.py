import sys
import pickle
import copy

from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer

class tester:
    features_and_tags = []
    testing_data = []

    def __init__(self, testing_data):
        self.testing_data = testing_data

    ################# feature #################
    # feature: abbreviation, e.g. CEO; Mr.
    def checkAbbreviation(self, token):
        if len(token[0]) < 2:  # e.g. 'I' is a special one
            return 0
        elif token[0][0].isupper() and token[0][-1] == '.':
            if len(token[0]) > 2:
                return 1
        return 0

    # feature: tag, if it is n. it has some possibility to be a title
    def checkTag(self, token):
        tag = token[1]
        if tag == "NN" or tag == "NNS" or tag == "NNP" or tag == "NNPS":
            return 1
        else:
            return 0

    # feature: affix e.g. director
    def checkAffix(self, token):
        affix_list = ['er', 'or', 'ist', 'an', 'ive', 'ant', 'man', 'ess']
        wnl = WordNetLemmatizer()
        word = wnl.lemmatize(token[0])
        if word[-2:] in affix_list or word[-3:] in affix_list:
            if self.checkTag(token) == 1:
                return 1
            else:
                return 0
        else:
            return 0

    # feature: previous word's tag
    def checkPreviousWord(self, sentence, index):
        if index == 0:
            return 0
        else:
            if self.checkTag(sentence[index - 1]) == 1:
                return 1
            else:
                return 0
    # feature: next word's tag
    def checkNextWord(self, sentence, index):
        if index == len(sentence) - 1:
            return 0
        else:
            if self.checkTag(sentence[index + 1]) == 1:
                return 1
            else:
                return 0

    # start to build feature_and_tags list
    def collectFeature(self):
        for sentence in self.testing_data:
            index = 0
            for token in sentence:
                feature_dict = {}  # used to store every word's feature
                ##### features #######
                feature_dict[token[0]] = 1  # taking the word itself as a feature
                feature_dict['previous_tag'] = self.checkPreviousWord(sentence, index)
                feature_dict['next_tag'] = self.checkNextWord(sentence,index)
                feature_dict['abbr'] = self.checkAbbreviation(token)
                feature_dict['Affix'] = self.checkAffix(token)
                ######################
                self.features_and_tags.append(feature_dict)
                index += 1

    ############################################

def F1_measure(testing_data, prediction_result):
    testing_real_result = []
    for sentence in testing_data:
        for token in sentence:
            if token[2] == 'O':
                testing_real_result.append(0)
            else:
                testing_real_result.append(1)
    # testing_real_result = [1]
    # prediction_result = [1]
    measure = f1_score(testing_real_result, prediction_result, pos_label=1, average='binary')
    return measure

if __name__ == '__main__':
    path_to_testing_data = sys.argv[1]
    path_to_classifier = sys.argv[2]
    path_to_result = sys.argv[3]
    # load testing data
    with open(path_to_testing_data, 'rb') as f:
        testing_data = pickle.load(f)
    # print (testing_data)
    # load classifier
    with open(path_to_classifier, 'rb') as f:
        classifier = pickle.load(f)

    # get feature list
    t = tester(testing_data)
    t.collectFeature()
    X = t.features_and_tags
    # print(X)
    with open("./vectorizer_data", 'rb') as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(X)
    # make prediction
    prediction_result = classifier.predict(X)

    # write output to the file
    output = []
    index = 0
    for sentence in testing_data:
        l = []
        for token in sentence:
            if prediction_result[index] == 1:
                l.append((token[0], 'TITLE'))
            else:
                l.append((token[0], 'O'))
            index += 1
        output.append(copy.deepcopy(l))

    print(F1_measure(testing_data, prediction_result))
    with open (path_to_result, 'wb') as f:
        pickle.dump(output, f)



