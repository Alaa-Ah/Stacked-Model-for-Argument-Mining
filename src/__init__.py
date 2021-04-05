import os

ARG_EXTRACTION_ROOT_DIR = os.path.abspath(os.getcwd())

models_dir = ARG_EXTRACTION_ROOT_DIR + '/models/'
output_dir = ARG_EXTRACTION_ROOT_DIR + '/results-output/'
news_output_dir = output_dir + 'stock-market-news'


if not os.path.exists(output_dir):
        os.mkdir(output_dir)

if not os.path.exists(models_dir):
    os.mkdir(models_dir)

if not os.path.exists(news_output_dir):
    os.mkdir(news_output_dir)
