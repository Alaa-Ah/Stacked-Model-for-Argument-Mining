from src.classical_ML.src.config import *
from src.classical_ML.src.data_reader import DataUnification, LoadEssaysSentences
from src.classical_ML.src.vectorizer import Vectorizer

from sklearn.model_selection import StratifiedKFold

data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

# DataUnification()
with open(data_dir + 'essays_sentences.json', encoding='utf-8') as f:
    sentences_all = json.load(f)

y = [sent['sent-class'] for sent in sentences_all]

tmp = []
for c in y:
    if c == 'c':
        tmp.append(1)
    elif c == 'p':
        tmp.append(2)
    else:
        tmp.append(0)
y = tmp

print('end parsing')

vectorizer = Vectorizer()
vectorizer.fit(sentences_all)

def train_svm():
    '''
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_idx, test_idx in kf.split(sentences_all, y):
        train_sentences = sentences_all[train_idx[0] : train_idx[-1] + 1]
        test_sentences = sentences_all[test_idx[0] : test_idx[-1] + 1]
        print('Start vectorization')
        y_train = [sent['sent-class']  for sent in train_sentences]
        y_test = [sent['sent-class']  for sent in test_sentences]
        x_vec_train = vectorizer.transform(train_sentences)
        x_vec_test = vectorizer.transform(test_sentences)
        print('End vectorization')
        svmClf = svm.SVC(kernel='rbf', C=107)
        svmClf.fit(x_vec_train, y_train)
        y_pred_svm = svmClf.predict_proba(x_vec_test)
        svm_acc = accuracy_score(y_test, y_pred_svm)
        print(f'Accuracy score on testing data: {svm_acc}')
    '''

    train_sentences = [sent for sent in sentences_all if sent['train']]
    test_sentences = [sent for sent in sentences_all if not sent['train']]

    print('Start vectorization')
    y_train = [sent['sent-class']  for sent in train_sentences]
    y_test = [sent['sent-class']  for sent in test_sentences]
    x_vec_train = vectorizer.transform(train_sentences)
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')

    svmClf = svm.SVC(kernel='rbf', C=107)
    scores = cross_val_score(svmClf, x_vec_train, y_train, cv=2, n_jobs=36)
    print(f'10-fold cross validation accuracy score: {scores.mean()}')

    svmClf.fit(x_vec_train, y_train)
    y_pred_svm = svmClf.predict(x_vec_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    print(f'Accuracy score on testing data: {svm_acc}')

def svm_train_and_predict(train_sentences, y_train, test_sentences):
    svmClf = svm.SVC(kernel='rbf', C=107, probability=True)
    print('Start vectorization')
    # y_train = [sent['sent-class']  for sent in train_sentences]
    y_test = [sent['sent-class']  for sent in test_sentences]
    x_vec_train = vectorizer.transform(train_sentences)
    x_vec_test = vectorizer.transform(test_sentences)
    print('End vectorization')
    svmClf.fit(x_vec_train, y_train)
    y_pred_svm = svmClf.predict_proba(x_vec_test)
    # svm_acc = accuracy_score(y_test, y_pred_svm)
    # print(f'Accuracy score on testing data: {svm_acc}')
    return svmClf, y_pred_svm.tolist()

def svm_predict(model, sentences, proba=True):
    x_vec = vectorizer.transform(sentences)
    if proba:
        preds = model.predict_proba(x_vec)
        return preds.tolist()

    preds = model.predict(x_vec)
    return preds.tolist()