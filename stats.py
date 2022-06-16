import os
import re
import json

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())
DATASET_ARG_QUAL_IN_DIR = ARG_EXTRACTION_ROOT_DIR + '/corpora/earning-calls/'

def get_list_of_files(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_files(fullPath)
        else:
            allFiles.append(fullPath)              
    return allFiles


#Custom inter-annotator agreement based on identical start and stop and argument type or relations between argument
def get_inter_annotator_agreement(ann_filenames):
	nb_annotations = 0
	nb_identical_annotations = 0
	nb_identical_relations = 0
	annotations_idx = set()
	annotations = set()
	identical_annotations = set()
	relations = set()
	identical_relations = set()
	for ann_filename in ann_filenames:
		with open(ann_filename, 'r') as ann_file:
			lines_ann = ann_file.readlines()
			for line in lines_ann:
				nb_annotations = nb_annotations + 1
				if len(line.split(' ')) >= 5: 
					id, type_label, start, end = line.split(' ')[0:4]
					if (type_label, start, end) in annotations:
						if (type_label, start, end) in identical_annotations:
							nb_identical_annotations = nb_identical_annotations + 1
						else:
							nb_identical_annotations = nb_identical_annotations + 2
						identical_annotations.add((type_label, start, end))
					annotations.add((type_label, start, end))
					annotations_idx.add((id, type_label, start, end))
				else:
					type_relation, arg1, arg2 = line.split(' ')[1:4]
					arg1 = arg1[5:].strip()
					arg2 = arg2[5:].strip()
					type1, start1, end1 = [(annotation[1], annotation[2], annotation[3]) for annotation in annotations_idx if annotation[0] == arg1][0]
					type2, start2, end2 = [(annotation[1], annotation[2], annotation[3]) for annotation in annotations_idx if annotation[0] == arg2][0]
					if (type_relation, type1, type2, start1, start2, end1, end2) in relations:
						if (type_relation, type1, type2, start1, start2, end1, end2) in identical_relations:
							nb_identical_annotations = nb_identical_annotations + 1
						else:
							nb_identical_annotations = nb_identical_annotations + 2
						identical_relations.add((type_relation, type1, type2, start1, start2, end1, end2))
					relations.add((type_relation, type1, type2, start1, start2, end1, end2))

	return (nb_identical_annotations + nb_identical_relations) / nb_annotations


def get_stats(company = None):
	nb_premises = 0
	nb_claims = 0
	nb_non_arg = 0
	nb_rel_support = 0
	nb_rel_attacks = 0
	nb_unlinked = 0
	nb_total_arg = 0
	nb_total_rel = 0

	percentage_premises = 0
	percentage_claims = 0
	percentage_non_arg = 0
	percentage_rel_support = 0
	percentage_rel_attacks = 0
	percentage_unlinked = 0


	nb_q_a_doc = 0
	nb_earning_calls_files = 0
	nb_answers = 0
	nb_questions = 0

	annotator_agreement = 0

	all_files = os.listdir(DATASET_ARG_QUAL_IN_DIR)

	if company is None:
		ann_texts = set([file[:file.index('.') - 2] for file in all_files if file.endswith(".ann")])
		json_texts = set([file for file in all_files if file.endswith(".json")])
	else:
		ann_texts = set([file[:file.index('.') - 2] for file in all_files if file.endswith(".ann") and file.startswith(company)])
		json_texts = set([file for file in all_files if file.endswith(".json") and file.startswith(company)])

	#to_remove = []
	#for json_file in json_texts:
	#	if not json_file.split('.')[0] + '_9.ann' in ann_texts:
	#		to_remove.append(json_file)
	#
	#if len(to_remove) ==0:
	#	print('problème : ' + company)
	#print('1 : ' + str(len(json_texts)))
	#json_texts.difference_update(to_remove)
	#print('2 : ' + str(len(json_texts)))

	nb_q_a_doc = len(json_texts)
	nb_earning_calls_files = len(set(['_'.join(file.split('_')[:3]) for file in ann_texts]))

	for ann in ann_texts:
		pattern = re.compile("{}_\d\.ann".format(ann))
		annotators = [file[file.index('.') - 1 : file.index('.')] for file in all_files 
						if pattern.match(file)]

		filenames_all_annotators = [DATASET_ARG_QUAL_IN_DIR + file for file in all_files if pattern.match(file)]
		if(len(filenames_all_annotators) > 1):
			annotator_agreement = annotator_agreement + get_inter_annotator_agreement(filenames_all_annotators)

		chosen_annotator = None
		if '9' in annotators:
			chosen_annotator = '9'
		else:
			chosen_annotator = annotators[0]
		
		ann_filename = ann + '_' + chosen_annotator + '.ann'
		with open(DATASET_ARG_QUAL_IN_DIR + ann_filename , 'r', encoding='utf-8') as ann_file:
			lines_ann = ann_file.readlines()
			ids = set()
			rel_used = set()
			for line in lines_ann:
				type = line.split(' ')[1]
				if len(line.split(' ')) >= 5 and type != 'NON-ARG': 
					ids.add(line.split(' ')[0])
				else:
					arg1, arg2 = line.split(' ')[2:4]
					arg1 = arg1[5:].strip()
					arg2 = arg2[5:].strip()
					rel_used.update([arg1, arg2])
				if 'PREMISE' in type:
					nb_premises = nb_premises + 1
					nb_total_arg = nb_total_arg + 1
				elif 'CLAIM' in type:
					nb_claims = nb_claims + 1
					nb_total_arg = nb_total_arg + 1
				elif 'NON-ARG' == type:
					nb_non_arg = nb_non_arg + 1
					nb_total_arg = nb_total_arg + 1
				elif 'SUPPORT' == type:
					nb_rel_support = nb_rel_support + 1
					nb_total_rel = nb_total_rel + 1
				elif 'ATTACK' == type:
					nb_rel_attacks = nb_rel_attacks + 1
					nb_total_rel = nb_total_rel + 1
			
			nb_unlinked = nb_unlinked + len(ids - rel_used)

	annotator_agreement = annotator_agreement / len(ann_texts)

	percentage_premises = (nb_premises / nb_total_arg) * 100
	percentage_claims = (nb_claims / nb_total_arg) * 100
	percentage_non_arg = (nb_non_arg / nb_total_arg) * 100
	percentage_rel_support = (nb_rel_support / nb_total_rel) * 100
	percentage_rel_attacks = (nb_rel_attacks / nb_total_rel) * 100
	percentage_unlinked = (nb_unlinked / (nb_premises + nb_claims)) * 100

	for json_filename in json_texts:
		with open(DATASET_ARG_QUAL_IN_DIR + json_filename , 'r', encoding='utf-8') as json_file:
			json_data = json.load(json_file)
			for annotation in json_data['annotations']:
				for result in annotation['result']:
					if 'ANSWER' in result['value']['labels']:
						nb_answers = nb_answers + 1
					elif 'QUESTION' in result['value']['labels']:
						nb_questions = nb_questions + 1

	string_to_add = ('_{}'.format(company) if company is not None else '')
	filepath = ARG_EXTRACTION_ROOT_DIR + '/statistics/statistics' + string_to_add + '.txt'
	
	with open(filepath, 'w') as result_file:
		if company is None:
			result_file.write("Global results (on the whole dataset) : \n")
		else:
			result_file.write("Results for the company {} : \n".format(company))
		
		result_file.write("""
	*LABELS STATS
		- PREMISES, CLAIMS and NON-ARG labels :
			Percentages are given relatively to the total number of premise claim and non-arg labels:
				Premises : #{}  ==> {}% 
				Claims :  #{}  ==> {}% 
				Non-Args : #{}  ==> {}% 
		
		- ATTACK and SUPPORT relations : 
			Percentages are given relatively to the total number of attack and support relations:
				Attack : #{}  ==> {}% 
				Support : #{}  ==> {}% 
		
		- UNLINKED Premises or Claims : 
			Percentage is given relatively to the total number of premise and claim labels:
				Unlinked label : #{}  ==> {}% 

	**OTHER STATS :
		Number of earning call files : #{}
		Number of question/answer files : #{}
		Total number of answers : #{}
		Total number of questions : #{}

        Custom Annotator agreement : {}

		""".format(nb_premises, percentage_premises, nb_claims, percentage_claims, nb_non_arg, percentage_non_arg, \
				nb_rel_attacks, percentage_rel_attacks, nb_rel_support, percentage_rel_support, nb_unlinked, percentage_unlinked, \
				nb_earning_calls_files, nb_q_a_doc, nb_answers, nb_questions, annotator_agreement))
		

companies = set()
docs = os.listdir(DATASET_ARG_QUAL_IN_DIR)
for doc in docs:
	if not doc.startswith('.'):
		c_name = str(doc.split('_')[0])
		companies.add(c_name)

for company_ in companies:
	print(company_)
	get_stats(company=company_)

get_stats()