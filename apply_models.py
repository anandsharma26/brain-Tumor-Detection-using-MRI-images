from sklearn.svm import SVC
import glcm_features as feat
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def applymodel(DataSet):
    Target = DataSet["Label"]
    Predictors = DataSet.drop(columns='Label')
    #print(Predictors)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf1 = make_pipeline(StandardScaler(), LogisticRegression())
    clf2 = make_pipeline(StandardScaler(), GaussianNB())


    X_train,X_test , y_train ,y_test = train_test_split(Predictors , Target ,test_size = .2)
    clP = clf.fit(X_train, y_train)

    print("Accuracy with SVM is  " , accuracy_score(y_test , clP.predict(X_test)))


    X_train,X_test , y_train ,y_test = train_test_split(Predictors , Target ,test_size = .2)
    clP = clf1.fit(X_train, y_train)

    print("Accuracy with LogisticRegression is  " , accuracy_score(y_test , clP.predict(X_test)))


    X_train,X_test , y_train ,y_test = train_test_split(Predictors , Target ,test_size = .2)
    clP = clf2.fit(X_train, y_train)

    print("Accuracy with GaussianNB is  " , accuracy_score(y_test , clP.predict(X_test)))

DataSet = feat.return_DatasetOriginal()
DataSet1 = feat.return_DatasetFiltered()

print("  ::::: Original Images results ::::: ")
applymodel(DataSet)
print("  ::::: Filtered Images results ::::: ")
applymodel(DataSet1)
