from src.classical_ML.src.config import *
from src.classical_ML.src.data_reader import ProcessRowText

source_dir = ARG_EXTRACTION_ROOT_DIR + '/src/classical_ML/src/'

claim_indicators = []
with open(source_dir + 'claim_indicators.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line: claim_indicators.append(line)

premise_indicators = []
with open(source_dir + 'premise_indicators.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line: premise_indicators.append(line)

keyword_indicators = []
with open(source_dir + 'key_words.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line: keyword_indicators.append(line)


def _GetParseTreeHeight(root):
    if not list(root.children):
        return 1
    else:
        return 1 + max(_GetParseTreeHeight(x) for x in root.children)

def _GetTenseFromTag(tag):
    if tag in ['VBD', 'VBN']:
        return 'pasteTense'
    if tag in ['VBG', 'VBP', 'VBZ']:
        return 'presenceTense'
    if tag == 'VB':
        return 'baseFormTense'
    if tag == 'MD':
        return 'modalVerb'
    return ''

def _GetWordsFrequency(sentence_text, words_list):
    words_count = 0
    for word in words_list:
        if word in sentence_text:
            words_count += 1
    return words_count

class Vectorizer():
    def __init__(self, params=None):
        print('Initializing vectorizer ...')
        if params:
            max_lemma_gram, max_pos_gram, lemma_min_df, pos_min_df = params

            self.lemma_vectorizer = CountVectorizer(binary=True, lowercase=True, stop_words='english', encoding='utf-8', ngram_range=(1, max_lemma_gram), min_df=lemma_min_df)
            self.pos_vectorizer = CountVectorizer(binary=True, encoding='utf-8', ngram_range=(1, max_pos_gram), min_df=pos_min_df)
            self.other_vectorizer = CountVectorizer(binary=True, encoding='utf-8')
        else:
            self.lemma_vectorizer = CountVectorizer(binary=True, lowercase=True, stop_words='english', encoding='utf-8', ngram_range=(1, 3), min_df=5)
            self.pos_vectorizer = CountVectorizer(binary=True, encoding='utf-8', ngram_range=(1, 2), min_df=3)
            self.other_vectorizer = CountVectorizer(binary=True, encoding='utf-8')

        self.x_pos = []
        self.x_lemma = []
        self.x_other = []

    def fit(self, sentences):
        print('Fitting ...')
        starting_time = time.time()

        self.x_pos, self.x_lemma, self.x_other = self._GetDocumentTermsMatrices(sentences)
        self.lemma_vectorizer.fit(self.x_lemma)
        self.pos_vectorizer.fit(self.x_pos)
        self.other_vectorizer.fit(self.x_other)

        print('time {} s'.format(time.time() - starting_time))

    def transform(self, sentences=None, row_text=None):
        starting_time = time.time()

        if sentences:
            print('Transforming a unified list of sentences ...')
            x_pos, x_lemma, x_other = self._GetDocumentTermsMatrices(sentences)

        elif row_text:
            print('Transforming a row text ...')
            row_text_sentences = processRowText(row_text)
            x_pos, x_lemma, x_other = self._GetDocumentTermsMatrices(row_text_sentences)

        else:
            print('Transforming all documents ...')
            x_pos, x_lemma, x_other = self.x_pos, self.x_lemma, self.x_other

        x_lemma_vec = self.lemma_vectorizer.transform(x_lemma)
        x_pos_vec = self.pos_vectorizer.transform(x_pos)
        x_other_vec = self.other_vectorizer.transform(x_other)
        x_vec = sparse.hstack((x_lemma_vec, x_pos_vec, x_other_vec), format='csr')
        x_vec_normalized = normalize(x_vec, norm='l1', axis=0)
        print('time {} s'.format(time.time() - starting_time))

        return x_vec

    def _GetDocumentTermsMatrices(self, sentences):
        x_pos, x_lemma, x_other = [], [], []
        for index, sentence in enumerate(sentences):
            features = self._GetSentenceFeatures(sentence)
            x_pos.append(' '.join(features['pos-array']))
            x_lemma.append(' '.join(features['lemma-array']))
            other = ''
            other += ('digitCount ' * features['digit-count'])
            other +=  ' '.join(features['entity-array']) + ' '
            other += ' '.join(features['tense-array']) + ' '
            other += ('parseTrHeight ' * features['parse-tree-depth'])
            other += ('tokenCount' * features['number-of-tokens'])
            other += ('keyWordsCount ' * features['key-words-count'])
            other += ('claimWordsCount' * features['claim-words-count'])
            other += ('premiseWordsCount' * features['premise-words-count'])
            # other += features['tagArray'] + ' '
            other += ('sentencePosition ' * features['sentence-position'])
            other += ('subclauseCount ' * features['number-of-subclauses'])
            # achieved accuracy
            # acc 0.6428072600852872, 0.7093275488069414
            # other += ('isFirstSentence ' * features['is-first-sentence'])
            # other += ('isLastSentence ' * features['is-last-sentence'])
            other += ('isInIntroduction ' * features['is-in-introduction'])
            other += ('isInConclusion ' * features['is-in-conclusion'])
            other += ('ponctuationCount ' * features['number-of-ponctuation-marks'])
            other += ('questionMarkEnding ' * features['question-mark-ending'])
            # achieved accuracy
            # acc 0.7513861287169139, 0.7939262472885033

            x_other.append(other.strip())

        return (x_pos, x_lemma, x_other)

    def _GetSentenceFeatures(self, sentence):

        doc = nlp(sentence['sent-text'])
        pos_s, tag_s, lemma_s, entity_s, digit_count, tense_s = [], [], [], [], [], []
        sub_clauses = digit_count = ponctuation_count = 0

        root = list(doc.sents)[0].root
        tree_height = _GetParseTreeHeight(root)
        tokens = list(doc)

        for index, token in enumerate(tokens):
            pos_s.append(token.pos_)
            tag_s.append(token.tag_)
            lemma_s.append(token.lemma_)

            token_tense = _GetTenseFromTag(token.tag_)
            if token_tense: tense_s.append(token_tense)

            if token.ent_type != 0: entity_s.append(token.ent_type_)
            if token.dep_ in ('xcomp', 'ccomp'): sub_clauses += 1
            if token.is_digit: digit_count += 1
            if 'PUNCT' == token.pos_: ponctuation_count += 1


        key_words_freq = _GetWordsFrequency(sentence['sent-text'], keyword_indicators)
        claim_words_freq = _GetWordsFrequency(sentence['sent-text'], claim_indicators)
        premise_words_freq = _GetWordsFrequency(sentence['sent-text'], premise_indicators)

        features = {
            ###########################
            # Structural features
            ###########################
            'is-first-sentence' : (0 == sentence['sent-idx'])
            ,'is-last-sentence' : sentence['is-last-sent']
            ,'is-in-introduction' : (0 == sentence['parag-idx'])
            ,'is-in-conclusion' : sentence['is-last-parag']
            ,'sentence-position' : sentence['sent-idx']
            ,'number-of-tokens' : len(tokens)
            ,'number-of-ponctuation-marks' : ponctuation_count
            ,'question-mark-ending' : ('?' == tokens[-1].text)
            ###########################
            # Lexical features
            ###########################
            ,'lemma-array' : lemma_s
            ,'pos-array' : pos_s
            ,'tense-array' : tense_s
            ,'entity-array' : entity_s
            ###########################
            # Syntactic features
            ###########################
            ,'parse-tree-depth' : tree_height
            ,'number-of-subclauses' : sub_clauses
            ###########################
            # Indicators
            ###########################
            ,'key-words-count' : key_words_freq
            ,'claim-words-count' : claim_words_freq
            ,'premise-words-count' : premise_words_freq
            ,'digit-count' : digit_count
        }
        return features
