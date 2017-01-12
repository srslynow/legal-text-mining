from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np

if __name__ == '__main__':
    rechtspraak_train_text = np.load("train_data.npy")
    rechtspraak_train_labels = np.load("train_label.npy")
    rechtspraak_test_text = np.load("test_data.npy")
    rechtspraak_test_labels = np.load("test_label.npy")

    print("Training classifier")
    classif = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=11), n_jobs=-1)
    classif.fit(rechtspraak_train_text, rechtspraak_train_labels)
    score_acc = classif.score(rechtspraak_test_text,rechtspraak_test_labels)
    joblib.dump(classif, 'model_knn.pkl')
    print("Score: " + str(score_acc))