import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import glob

# path variables
xmls_path = r'E:\Sync\Datasets\rechtspraak\txt'
files = glob.glob(os.path.join(xmls_path, '*.txt'))

# vars to be filled
rechtspraak_text = []
rechtspraak_labels = []

# use labels
use_labels = ['Strafrecht','Vreemdelingenrecht','Socialezekerheidsrecht','Belastingrecht','Civiel recht','Bestuursrecht','Personen- en familierecht']

print("Reading clean text files")

for i,file in enumerate(files):
    if i % 1000 == 0:
        print(str(i) + '\t\t' + file)
    with open(file) as txt_file:
        subjects = txt_file.readline().strip('\n').split(',')
        clean_text = txt_file.readline().strip('\n')
        # save to list
        rechtspraak_labels.append(subjects)
        rechtspraak_text.append(clean_text)

# filter labels
rechtspraak_labels = [[label for label in lab_row if label in use_labels] for lab_row in rechtspraak_labels]
# delete rows with no labels
delete_indices = [i for i,lab_row in enumerate(rechtspraak_labels) if len(lab_row) == 0]
for i in delete_indices:
    del rechtspraak_labels[i]
    del rechtspraak_text[i]
print("Encoding & binarizing labels")
mlb = MultiLabelBinarizer()
#mlb.fit(set(use_labels))
rechtspraak_labels = mlb.fit_transform(rechtspraak_labels)
# split into training & test set
train_test_split = np.random.choice([0, 1], size=len(rechtspraak_labels), p=[.75, .25])
rechtspraak_train_labels = [rechtspraak_labels[elem] for elem,i in enumerate(train_test_split) if i == 0]
rechtspraak_test_labels = [rechtspraak_labels[elem] for elem,i in enumerate(train_test_split) if i == 1]
rechtspraak_train_text = [rechtspraak_text[elem] for elem,i in enumerate(train_test_split) if i == 0]
rechtspraak_test_text = [rechtspraak_text[elem] for elem,i in enumerate(train_test_split) if i == 1]

print("Building vocabulary")
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
rechtspraak_train_text = vectorizer.fit_transform(rechtspraak_train_text).toarray()
rechtspraak_test_text = vectorizer.transform(rechtspraak_test_text).toarray()

np.save("train_data", rechtspraak_train_text)
np.save("train_label", rechtspraak_train_labels)
np.save("test_data", rechtspraak_test_text)
np.save("test_label", rechtspraak_test_labels)

print("Writing vocabulary to file")
with open("vocabulary.txt", 'w') as vocabulary_txt:
    vocabulary = vectorizer.get_feature_names()
    for vocab in vocabulary:
        vocabulary_txt.write(vocab + '\n')