from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import numpy as np

model = joblib.load('model_knn.pkl')
X_test = np.load("test_data.npy")
y_test = np.load("test_label.npy")
y_out = model.predict(X_test)
y_out[y_out < 0.5] = 0
y_out[y_out >= 0.5] = 1
corr = y_out * y_test
gt = np.sum(y_test, axis=0)
pred = np.sum(corr, axis=0)
perc = np.divide(pred, gt) * 100
print("Ground truth", gt)
print("Predicted correct", pred)
print("Percentage correct", perc)