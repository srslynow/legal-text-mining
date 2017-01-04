import os
import sys
import glob
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import re
import string
import urllib2

stopWords = stopwords.words("dutch")
tagsRegex = re.compile(r'<[^<]+?>')
spacesRegex = re.compile(r'[\s][\s]+')
bracket_regex = re.compile('[(|)|\[|\]]')
punctuationRegex = re.compile('[%s]' % re.escape(
    string.punctuation))  # remove punctuation

xml_text = urllib2.urlopen(
    "http://www.rvo.nl/sites/default/files/open_data/dop_nieuws.xml").read()
try:
    tree = ET.fromstring(xml_text)
except:
    print "Could not parse"

for i, bericht in enumerate(tree):
    berichttext = bericht.find('Berichttekst').text
    rawText = berichttext
    rawText = tagsRegex.sub('', rawText)
    rawText = rawText.encode('ascii', errors='ignore')
    rawText = string.lower(rawText)
    rawText = spacesRegex.sub(' ', rawText).strip()
    rawText = rawText.replace('[', '').replace(']', '')
    rawText = doubleBracketRegex.sub('', rawText)
    cleanASCII = ' '.join(word for word in rawText.split()
                          if word not in stopWords)

    txtFile = open(os.path.join(
        '/media/koen/Sync/Datasets/rechtspraak/txt/negative', str(i) + '.txt'), 'w')
    txtFile.write(cleanASCII)
    txtFile.close()
