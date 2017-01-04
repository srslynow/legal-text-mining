import os
import sys
import glob
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import re
import string


# path variables
xmls_path = r'E:\Datasets\rechtspraak\xml_positive'
files = glob.glob(os.path.join(xmls_path, '*.xml'))

# vars to be filled
subject_counter = {}

# xml namespaces
ns = {
    "rechtspraakns": "http://www.rechtspraak.nl/schema/rechtspraak-1.0",
    "dcterms": "http://purl.org/dc/terms/"
    }

# regexes for text cleaning purposes
stopwords = stopwords.words("dutch")
spaces_regex = re.compile('[\s][\s]+')
bracket_regex = re.compile('[(|)|\[|\]]')
punctuation_regex = re.compile('[%s]' % re.escape(string.punctuation))
ascii_regex = re.compile(r'[^\x00-\x7F]+')

for i,file in enumerate(files):
    if i % 100 == 0:
        print(str(i) + '\t\t' + file)
    try:
        tree = ET.parse(file)
    except:
        continue
    # get filename
    fname = os.path.splitext(os.path.basename(file))[0]
    # get xml elements we need
    uitspraak_elem = tree.find('rechtspraakns:uitspraak', ns)
    subject_elem = tree.find('.//dcterms:subject', ns)
    # set subjects and do some sanity checks
    if uitspraak_elem is None or subject_elem is None: continue
    subjects = [x.strip() for x in subject_elem.text.split(';')]
    for subj in subjects: subject_counter[subj] = subject_counter.setdefault(subj, 0) + 1
    if len(subjects) == 0: continue
    # text cleaning
    raw_text = ' '.join(uitspraak_elem.itertext())
    raw_text = raw_text.lower()
    raw_text = spaces_regex.sub(' ', raw_text).strip()
    raw_text = bracket_regex.sub('', raw_text)
    raw_text = punctuation_regex.sub('', raw_text)
    raw_text = ' '.join(word for word in raw_text.split() if word not in stopwords and word.isalpha())
    raw_text = ascii_regex.sub('', raw_text)
    # save to file
    txt_file = open(os.path.join('txt', fname + '.txt'), 'w')
    txt_file.write(','.join(subjects) + '\n')
    txt_file.write(raw_text)
    txt_file.close()

print("Finished preprocessing all xml files")
print("Subject counts:")
print(subject_counter)

