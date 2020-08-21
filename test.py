import argparse
from ast import literal_eval

parser = argparse.ArgumentParser(description='Update')
parser.add_argument('keywords_txt', help= 'keywords_txt file')
args = parser.parse_args()

keywords = literal_eval(open(args.keywords_txt, encoding='utf-8').read())
print(keywords)

