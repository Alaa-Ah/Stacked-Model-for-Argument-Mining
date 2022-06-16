from src.transformers_ML.config import *
from src.transformers_ML.utils import *
from src.transformers_ML.data_loader import DataLoadHandler
from src.transformers_ML.model_handler import *
from src.data_parser.utils import ParseEarningCalls

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# svm dependencies
from src.classical_ML.model_handler import *
from src.classical_ML.src.data_reader import ProcessRowSentences

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def premiseClaimMap(y):
    tmp = []
    for c in y:
        if c == 'c':
            tmp.append(1)  # claim
        elif c == 'p':
            tmp.append(0)  # premise
    return tmp


def GetTrainTestLists(x, y, fold=5):
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []

    test_data_length = len(x) // fold
    for index in range(fold):

        if (index + 1) * test_data_length <= len(x):
            x_test = x[index * test_data_length: (index + 1) * test_data_length]
            y_test = y[index * test_data_length: (index + 1) * test_data_length]
            x_train = x[: index * test_data_length] + x[(index + 1) * test_data_length:]
            y_train = y[: index * test_data_length] + y[(index + 1) * test_data_length:]
        else:
            x_test = x[index * test_data_length:]
            y_test = y[index * test_data_length:]
            x_train = x[: index * test_data_length]
            y_train = y[: index * test_data_length]

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        print(len(x_train), len(x_test), len(y_train), len(y_test))

    return x_train_list, x_test_list, y_train_list, y_test_list


def StackSvmBert():
    data_dir = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'

    with open(data_dir + 'essays_webd_ibm_p_c_with_prompt.json', encoding='utf-8') as f:
        sentences_others = json.load(f)
        y_others = premiseClaimMap([sent['sent-class'] for sent in sentences_others])  # sent class could be premise/ claim
        sentences_others = ProcessRowSentences([sent['sent-text'] for sent in sentences_others])
    
    with open(data_dir + 'earning-calls_sentences.json', encoding='utf-8') as f:
        sentences_earning_call = json.load(f)
        sentences_earning_call_p_c = [sent for sent in sentences_earning_call \
			if (sent['sent-class'] == 'c' or sent['sent-class'] == 'p')]
		
        y_ec = premiseClaimMap([sent['sent-class'] for sent in sentences_earning_call_p_c])  # sent class could be premise/ claim
        sentences_earning_call_p_c = ProcessRowSentences([sent['sent-text'] for sent in sentences_earning_call_p_c])
    
    sentences_all = sentences_others + sentences_earning_call_p_c
    y = y_others + y_ec

    temp_text, test_text, temp_labels, test_labels = train_test_split(sentences_all, y,
                                                                      random_state=2018,
                                                                    shuffle=True,
                                                                      test_size=0.20,
                                                                      stratify=y)

    print('training bert and svm all sentences...')
    bert_model, bert_preds = bert_train_and_predict(temp_text, temp_labels, test_text)
    svm_model, svm_preds = svm_train_and_predict(temp_text, temp_labels, test_text)

    meta_fold = 5
    x_train_list, x_test_list, y_train_list, y_test_list = GetTrainTestLists(temp_text, temp_labels, meta_fold)

    x_meta_train, y_meta_train = [], []
    for index in range(meta_fold):
        print(f'meta model data processing: fold {index + 1}')

        train_text, val_text, train_labels, val_labels = x_train_list[index], x_test_list[index], y_train_list[index], \
                                                         y_test_list[index]
        print(len(train_text), len(val_text), len(train_labels), len(val_labels))

        bert_model, y_bert = bert_train_and_predict(train_text, train_labels, val_text)
        svm_model, y_svm = svm_train_and_predict(train_text, train_labels, val_text)
        x_meta_train += [y_svm[idx][:1] + y_bert[idx] for idx in range(len(val_text))]
        y_meta_train += val_labels
        print(len(x_meta_train), len(y_meta_train))

    print('finish meta train data preparation')
    meta_model = LogisticRegression()
    meta_model.fit(x_meta_train, y_meta_train)

    # testing the performance of the whole pipline
    x_meta_test = [svm_preds[idx][:1] + bert_preds[idx] for idx in range(len(test_text))]
    meta_preds = meta_model.predict(x_meta_test)

    print("***test evaluation***")
    print("***classification_report for stacked model on essays_sentences***")
    print(classification_report(test_labels, meta_preds, digits=4))
    print("***classification_report for SVM model essays_sentences***")
    print(classification_report(test_labels, [int(pred[0] < pred[1]) for pred in svm_preds], digits=4))
    print("***classification_report for bert model essays_sentences***")
    print(classification_report(test_labels, [int(pred[0] < pred[1]) for pred in bert_preds], digits=4))

    print('#' * 50)
    print('Confusion matrices')

    svm_preds_labels = [int(pred[0] < pred[1]) for pred in svm_preds]
    bert_preds_labels = [int(pred[0] < pred[1]) for pred in bert_preds]

    print("***confusion_matrix for stacked model***")
    print(confusion_matrix(test_labels, meta_preds))
    print(confusion_matrix(test_labels, meta_preds,normalize='true'))
    print("***confusion_matrix for SVM model***")
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in svm_preds]))
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in svm_preds],normalize='true'))
    print("***confusion_matrix for bert model***")
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in bert_preds]))
    print(confusion_matrix(test_labels, [int(pred[0] < pred[1]) for pred in bert_preds],normalize='true'))


def main(args):
    print('args: ', args)
    action, obj = args
    print('action: ', action)
    print('obj: ', obj)
    if 'train' == action:
        if 'bert' == obj:
            print('Running BERT training...')
            train_model(epochs=3)
        elif 'svm' == obj:
            print('Running SVM training...')
            train_svm()
        elif 'stack' == obj:
            StackSvmBert()

    elif 'parse' == action:
        print('Parsing Earning Calls...')
        ParseEarningCalls()
    else:
        print('wrong params has been given!')


if __name__ == '__main__':
    main(sys.argv[1:])