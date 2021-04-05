import os
import time
import en_core_web_sm
import csv
import json
import string

nlp = en_core_web_sm.load()

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
DATASET_BASE_IN_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/essays/'
DATASET_BASE_OUT_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/parsed-corpora/'


# total claims: 2257
# total premises: 3832

def _GetEssaysFiles():
    essays = []
    with open(DATASET_BASE_IN_DIR + 'train-test-split.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                line_count += 1
                essay = {
                        'id':row[0]
                        ,'txt': row[0] + '.txt'
                        ,'ann': row[0] + '.ann'
                        ,'train': ('TRAIN' == row[1])
                        }
                essays.append(essay)
    return essays

def _ProcessEssay(essay):
    annotated_sentences = []
    
    file_txt = open(DATASET_BASE_IN_DIR + 'brat-project-final/' + essay['txt'], 'r', encoding='utf-8')
    file_ann = open(DATASET_BASE_IN_DIR + 'brat-project-final/' + essay['ann'], 'r', encoding='utf-8')
    try:
        # reading annotations
        claims_list = []
        premises_list = []
        lines_ann = file_ann.readlines()
        for line in lines_ann:
            if 'T' != line[0]: continue
            line = line.lower()
            arg_id, arg_info, arg_text = line.split('\t')[:3]
            arg_type, start, end = arg_info.split()
            if 'claim' in arg_type:
                claims_list.append(arg_text.strip())

            elif 'premise' in arg_type:
                premises_list.append(arg_text.strip())

        def _GetSentenceClass(sentence):
            global found_premises, found_claims, essay_claim_fnd, essay_premise_fnd
            for p in premises_list:
                if p in sentence:
                    return 'p'
            for c in claims_list:
                if c in sentence:
                    return 'c'
            return 'n'

        # reading original text
        full_text = file_txt.read().lower()
        full_text_splitted = full_text.split('\n\n')
        if 1 == len(full_text_splitted):
            full_text = full_text.replace('\n \n', '\n\n').replace('\n  \n', '\n\n')
            full_text_splitted = full_text.split('\n\n')

        topic, essay_text = full_text_splitted[:2]
        paragraphs = essay_text.split('\n')

        for para_idx, paragraph in enumerate(paragraphs):
            is_last_parag = (para_idx == len(paragraphs) - 1)
            document = nlp(paragraph)
            sentences_str = [sent.text for sent in document.sents]
            # sentences_str = paragraph.split('.')
            # sentences_str = [sent + '.' for sent in sentences_str if sent]

            for sent_idx, sent in enumerate(sentences_str):
                is_last_sent = (sent_idx == len(sentences_str) - 1)
                sent_class = _GetSentenceClass(sent)
                sent = sent.replace('\n', ' ')
                sentence_dict = {
                        'sent-text' : sent.strip()
                        ,'essay-id' : essay['id']
                        ,'parag-idx' : para_idx
                        ,'sent-idx' : sent_idx
                        ,'sent-class' : sent_class
                        ,'train' : int(essay['train'])
                        ,'is-last-parag' : is_last_parag
                        ,'is-last-sent' : is_last_sent
                    }
                annotated_sentences.append(sentence_dict)
    except Exception as e:
        print('Error occured while reading: {}\nError: {}'.format(essay['id'], e))

    finally:
        file_txt.close()
        file_ann.close()
        return annotated_sentences

def ProcessRowText(row_text):
    annotated_sentences = []
    text = row_text.lower()
    paragraphs = text.split('\n')
    for para_idx, paragraph in enumerate(paragraphs):
        is_last_parag = (para_idx == len(paragraphs) - 1)
        document = nlp(paragraph)
        sentences_str = [sent.text for sent in document.sents]

        for sent_idx, sent in enumerate(sentences_str):
            is_last_sent = (sent_idx == len(sentences_str) - 1)
            sent_class = 'u' # still unknown sentence class
            sent = sent.replace('\n', ' ')
            sentence_dict = {
                    'sent-text' : sent.strip()
                    ,'essay-id' : essay['id']
                    ,'parag-idx' : para_idx
                    ,'sent-idx' : sent_idx
                    ,'sent-class' : sent_class
                    ,'train' : 0
                    ,'is-last-parag' : is_last_parag
                    ,'is-last-sent' : is_last_sent
                }

            annotated_sentences.append(sentence_dict)
    return annotated_sentences

def DataUnification():
    print('start essays processing ...')

    essays = _GetEssaysFiles()

    starting_time = time.time()

    sentences_all = []
    for essay in essays:
        sentences = _ProcessEssay(essay)
        sentences_all += sentences

    # save all sentences
    print('saving all sentences...')

    with open(DATASET_BASE_OUT_DIR + 'essays_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(sentences_all, f)

def LoadEssaysSentences(file_name=DATASET_BASE_OUT_DIR + 'essays_sentences.json'):
    print('Loading annotated sentences ...')
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            sentences_all = json.load(f)
    except Exception as e:
        print('Error: {}'.format(e))
        sentences_all = []

    return sentences_all
