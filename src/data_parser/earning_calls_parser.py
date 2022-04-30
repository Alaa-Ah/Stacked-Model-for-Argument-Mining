import os
import time
import en_core_web_sm
import csv
import json
import string
import re

nlp = en_core_web_sm.load()

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
DATASET_BASE_IN_DIR = ARG_EXTRACTION_ROOT_DIR + '/../../corpora/earning-calls/'
DATASET_BASE_OUT_DIR = ARG_EXTRACTION_ROOT_DIR + '/../../corpora/parsed-corpora/'


# total claims: 
# total premises: 

def _GetEarningCallsFiles():
    earningCalls = []
    all_files = os.listdir(DATASET_BASE_IN_DIR)
    texts = set([file[:file.index('.') - 2] for file in all_files if file.endswith(".ann")])
    for text in texts:
        pattern = re.compile("{}_\d\.ann".format(text))
        annotators = [file[file.index('.') - 1 : file.index('.')] for file in all_files 
                      if pattern.match(file)]
        chosen_annotator = 9 if 9 in annotators else annotators[0]
        question_answer_text = ""
        answers = []
        with open(DATASET_BASE_IN_DIR + text + ".json") as json_file: 
            json_data = json.load(json_file)
            question_answer_text = json_data['data']['my_text']
            for result in json_data['annotations'][0]['result']:
                if result['value']['labels'][0] == 'ANSWER':
                    answers.append({'text': result['value']['text'],
                    'start': result['value']['start'],
                    'end': result['value']['end']
                    })
        if len(answers) ==0:
            print("pas d'answer pour le fichier {}".format(text))
        annotation = {
            'id': text + '_' + chosen_annotator,
            'answers': answers,
            'txt': question_answer_text,
            'ann': text + '_' + chosen_annotator + '.ann'
        }
        earningCalls.append(annotation)
    return earningCalls

def _ProcessEarningCallQA(question_answer):
    annotated_sentences = []
    file_ann = open(DATASET_BASE_IN_DIR + question_answer['ann'], 'r', encoding='utf-8')
    print(question_answer['id'])
    try:

        # reading annotations
        lines_ann = file_ann.readlines()
        for line in lines_ann:
            if len(line.split(' ')) <= 5: continue
            arg_id, arg_type, start, end = line.split(' ')[:4]
            if 'PREMISE' in arg_type:
                arg_type = 'p'
            elif 'CLAIM' in arg_type:
                arg_type = 'c'
            elif 'NON-ARG' == arg_type:
                continue

            start = int(start)
            end = int(end)
            arg_text = ' '.join(line.split(' ')[4:]).strip()
            answer = [answer['text'] for answer in question_answer['answers'] 
                if start >= answer['start'] and start < answer['end'] ][0]
            if answer[:2] == ": ":
                answer = answer[2:]
            elif answer[0] == " ":
                answer = answer[1:]
            #parag_idx = len(question_answer['txt'][:start].split('\n')) - 1
            #is_last_parag = (parag_idx == len(question_answer['txt'].split()) - 1)
            #document = nlp(answer)
            #sentences_str = [sent.text for sent in document.sents]
            #sent_idx = -1
            #for idx, sent_text in enumerate(sentences_str):
            #    if arg_text in sent_text:
            #        sent_idx = idx
            #        break
            #if sent_idx == -1:
            #    print("Erreur : " + arg_id, arg_type, start, end, '\n', arg_text, '\n', answer, '\n\n====FIN ERREUR=====\n\n' )
            #is_last_sent = (sent_idx == len(sentences_str) - 1)
            #[idx for idx, parag in enumerate(paragraphs) if arg_text in parag][0]
            #print(arg_id, arg_type, start, end, '\n', arg_text, '\n', answer, '\n====NVX ARG=====\n')
            arg_dict = {
                            'sent-text' : arg_text
                            ,'earning-call-id' : question_answer['id']
                            ,'sent-class' : arg_type
                            ,'start' : start
                            ,'end' : end
                            ,'text' : answer
                        }
            annotated_sentences.append(arg_dict)

    except Exception as e:
        print('Error occured while reading: {}\nError: {}'.format(question_answer['id'], e))

    finally:
        file_ann.close()
        return annotated_sentences

def DataUnification():
    print('start essays processing ...')

    earning_calls = _GetEarningCallsFiles()

    starting_time = time.time()

    sentences_all = []
    for question_answer in earning_calls:
        sentences = _ProcessEarningCallQA(question_answer)
        sentences_all += sentences

    # save all sentences
    print('saving all sentences...')

    with open(DATASET_BASE_OUT_DIR + 'earning-calls_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(sentences_all, f)


def LoadEarningCallsSentences(file_name=DATASET_BASE_OUT_DIR + 'earning-calls_sentences.json'):
    print('Loading annotated sentences ...')
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            sentences_all = json.load(f)
    except Exception as e:
        print('Error: {}'.format(e))
        sentences_all = []

    return sentences_all

DataUnification()