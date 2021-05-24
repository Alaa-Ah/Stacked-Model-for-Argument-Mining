from src.classical_ML.src.config import *

source_dir = ARG_EXTRACTION_ROOT_DIR + '/src/classical_ML/src/'

DATASET_BASE_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/essays/'


# total claims: 2257                                                                                                     │
# total premises: 3832

def _GetEssaysFiles():
    essays = []
    with open(DATASET_BASE_DIR + 'train-test-split.csv') as csv_file:
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
    
    file_txt = open(DATASET_BASE_DIR + 'brat-project-final/' + essay['txt'], 'r', encoding='utf-8')
    file_ann = open(DATASET_BASE_DIR + 'brat-project-final/' + essay['ann'], 'r', encoding='utf-8')
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
            for c in claims_list:
                if c in sentence:
                    return 'c'
            for p in premises_list:
                if p in sentence:
                    return 'p'
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
            sentences_str = [sent.text for sent in document.sents if len(sent.text) > 15]
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
                    ,'essay-id' : 'non-id' #essay['id']
                    ,'parag-idx' : para_idx
                    ,'sent-idx' : sent_idx
                    ,'sent-class' : sent_class
                    ,'train' : 0
                    ,'is-last-parag' : is_last_parag
                    ,'is-last-sent' : is_last_sent
                }

            annotated_sentences.append(sentence_dict)
    return annotated_sentences

def ProcessRowSentences(row_sentences):
    annotated_sentences = []
    for sent in row_sentences:
        is_last_sent = 0
        is_last_parag = 0
        para_idx = 0
        sent_idx = 0
        sent_class = 'u' # still unknown sentence class
        sent = sent.replace('\n', ' ')
        sentence_dict = {
                'sent-text' : sent.strip().lower()
                ,'essay-id' : 'non-id'
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

    with open(DATASET_BASE_DIR + '../parsed-corpora/essays_sentences.json', 'w', encoding='utf-8') as f:
        json.dump(sentences_all, f)

def LoadEssaysSentences(file_name=DATASET_BASE_DIR + '../annotated_sentences_all.pkl'):
    print('Loading annotated sentences ...')
    try:
        with open(file_name, 'rb') as f:
            sentences_all = pickle.load(f)
    except Exception as e:
        print('Error: {}'.format(e))
        sentences_all = []

    return sentences_all
