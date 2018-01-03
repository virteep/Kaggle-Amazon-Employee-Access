


import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import f1_score
from sklearn import preprocessing, model_selection, metrics, linear_model
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


def read_dataSet():
    """
    This function reads the training dataset
    :return: data: return all the features except the target
             target: return the target values for all the data points
    """
    data = np.loadtxt(open("train.csv"), delimiter=',', usecols=range(1, 9), skiprows=1)
    target = np.loadtxt(open("train.csv"), delimiter=',', usecols=[0], skiprows=1)
    return data, target


def oneHotEncoding():
    """
    This function performs one hot encoding on the raw data
    :return: data: the modified dataset
             target: the target values for each sample data point
    """
    data = np.loadtxt(open("train.csv"), delimiter=',', usecols=range(1, 9), skiprows=1)
    target = np.loadtxt(open("train.csv"), delimiter=',', usecols=[0], skiprows=1)
    #performs one hot encoding, fit and transforms the data
    obj = preprocessing.OneHotEncoder()
    obj.fit(data)
    data = obj.transform(data)
    return data, target

def featureSelection(data, target):
    """
    This function performs feature selection on data obtained after one hot encoding
    Uses Chi-Squared test for feature selection
    :param data: the dataset after one-hot encoding
    :param target: the target values for all the data points after one-hot encoding
    :return: modifiedData : dataset after feature selection
    """

    #selects 550 best attributes using chi-squared test
    selectBest_attribute = SelectKBest(chi2, k= 550)
    #fit and transforms the data
    selectBest_attribute.fit(data,target)
    modifiedData = selectBest_attribute.transform(data)
    return modifiedData

def roc_value(Y_test, prediction, fpr_score, tpr_score, mean_auc, roc_auc_value):
    """
    This function calculates fpr, tpr and AUC curve value for any algorithm
    :param Y_test: the actual values of the target class
    :param prediction: the predicted values of the target class
    :param fpr_score: the false positve rate
    :param tpr_score: the true positive rate
    :param mean_auc: the mean value of AUC curve for 10 folds
    :param roc_auc_value: each auc curve value across 10 folds
    :return: roc_auc_value: each auc curve value across 10 folds
    """

    #calculates fpr and tpr values for each model
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, prediction)
    fpr_score.append(fpr)
    tpr_score.append(tpr)

    #calculates auc for each model
    roc_auc = metrics.auc(fpr, tpr)
    mean_auc = mean_auc + roc_auc
    roc_auc_value.append("{0:.2f}".format(roc_auc))

    return fpr_score, tpr_score, roc_auc_value, mean_auc

def plotGraph():
    """
    This function sets lables for X and Y axis
    :return: None
    """
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("AUC comparison for all models.")
    plt.grid(True)
    plt.show()

def Naive_Bayes(modifiedData, target):
    """
    This function implements Naive Bayes algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #BernoilliNB creates Niave Bayes object
    bernoilli_nb = BernoulliNB(alpha = 1)  # alpha = 1 performs laplace estimation, to handle zero frequency problem'

    #variable initialization
    random_seed = 31  #random seed value
    kFold = 10        #for 10 fold cross validation

    #stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []

    #performs 10 fold cross validation
    for fold in range(kFold):

        #uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(modifiedData, target, test_size=.20,
                                                                        random_state= fold * random_seed)
        #fits and predicts the values
        bernoilli_nb.fit(X_train, Y_train)
        prediction = bernoilli_nb.predict_proba(X_test)[:,1]
        tempPrediction = bernoilli_nb.predict(X_test)
        #calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc, roc_auc_value)

    print("Mean AUC for Naive Bayes: %f" % (mean_auc / kFold))

    #plots AUC graph for Naive Bayes
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color = 'g', label = 'Naive Bayes')
    plt.legend(loc = 'upper left')

def Logistic_Regression(data, target):
    """
    This function implements Logistic Regression algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #creates object for logistic regression
    log_regrresion = linear_model.LogisticRegression(C=3)

    # variable initialization
    random_seed = 42   #random seed value
    kFold = 10         #for 10 fold cross validation

    #stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20, random_state= fold * random_seed)

        # fits and predicts the values
        log_regrresion.fit(X_train, Y_train)
        prediction = log_regrresion.predict_proba(X_test)[:,1]
        tempPrediction = log_regrresion.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc, roc_auc_value)

    print("Mean AUC Logistic Regression: %f" % (mean_auc/kFold))

    # plots AUC graph for Logistic Regression
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color = 'm', label = 'Logistic Regression')
    plt.legend(loc = 'upper left')

def Random_Forest(data, target):
    """
    This function implements Random Forest algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #creates object for random forest classifier
    random_forest = RandomForestClassifier(n_jobs= 10)

    # variable initialization
    random_seed = 31 #random seed value
    kFold = 10      #for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20,random_state= fold * random_seed)

        # fits and predicts the values
        random_forest.fit(X_train, Y_train)
        prediction = random_forest.predict_proba(X_test)[:, 1]
        tempPrediction = random_forest.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,roc_auc_value)

    print("Mean AUC Random Forest: %f" % (mean_auc / kFold))

    # plots AUC graph for Random Forest
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='r', label='Random Forest')
    plt.legend(loc='upper left')


def KNN(data, target):
    """
    This function implements KNN algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #KNN object using 7 neighbors
    kneighbor = KNeighborsClassifier(n_neighbors=7)

    # variable initialization
    ramdom_seed = 42  #random seed value
    kFold = 10        #for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20, random_state= fold * ramdom_seed)

        # fits and predicts the values
        kneighbor.fit(X_train, Y_train)
        prediction = kneighbor.predict_proba(X_test)[:, 1]
        tempPrediction = kneighbor.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc,roc_auc_value)

    print("Mean AUC KNN: %f" % (mean_auc / kFold))

    # plots AUC graph for KNN
    max_roc = roc_auc_value.index(max(roc_auc_value))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='b', label='KNN')
    plt.legend(loc='upper left')

def SVM(data, target):
    """
    This function implements SVM algorithm and plots the AUC graph
    :param modifiedData: the dataset
    :param target: the target class value
    :return: None
    """

    #creates object for linear SVM
    linearSVM = LinearSVC(penalty = 'l1',  random_state= 37, max_iter=1000, dual=False, C=3)

    #initializing the variable
    random_seed = 42    #random seed value
    kFold = 10          #for 10 fold cross validation

    # stores fpr, tpr, auc for each fold
    mean_auc = 0
    fpr_score = []
    tpr_socre = []
    roc_auc_value = []

    # performs 10 fold cross validation
    for fold in range(kFold):
        # uses train test split of ratio 80:20 to perform 10 fold cross validation
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=.20, random_state= fold * random_seed)

        # fits and predicts the values
        linearSVM.fit(X_train, Y_train)
        prediction = linearSVM.predict(X_test)

        # calculates fpr and tpr, auc values
        fpr_score, tpr_score, roc_auc_value, mean_auc = roc_value(Y_test, prediction, fpr_score, tpr_socre, mean_auc, roc_auc_value)

    print("Mean AUC for SVM: %f" % (mean_auc / kFold))

    # plots AUC graph for SVM
    max_roc = roc_auc_value.index("{0:.2f}".format(mean_auc / kFold))
    plt.plot(fpr_score[max_roc], tpr_socre[max_roc], color='y', label='SVM')
    plt.legend(loc='upper left')



rawData, target = read_dataSet()

#one hot encoding
data, target = oneHotEncoding()

"""print(" ***************Results using One hot encoding ***********************\n")
modifiedData = featureSelection(data, target)
start_time = time.time()
Naive_Bayes(data, target)
print("Time Required for Naive Bayes in sec:",  (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
Logistic_Regression(data,target)
print("Time Required for Logistic regression in sec: ", (time.time() - start_time))
print("------------------------------------------------------------------------------")

start_time = time.time()
Random_Forest(data, target)
print("Time Required for Random Forest in sec: " ,  (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
KNN(data, target)
print(" Time Required for KNN in sec: " , (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
SVM(data, target)
print("Time Required for SVM in sec: " ,(time.time() - start_time))
print("----------------------------------------------------------------------")
plotGraph()
"""

#one hot encoding + feature selection
modifiedData = featureSelection(data, target)
print(" ***************Results using One hot encoding & Feature selection***********************\n")
start_time = time.time()
Naive_Bayes(modifiedData, target)
print(" Time Required for Naive Bayes in sec:",  (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
Logistic_Regression(modifiedData,target)
print(" Time Required for Logistic regression in sec: " , (time.time() - start_time))
print("------------------------------------------------------------------------------")

start_time = time.time()
Random_Forest(modifiedData, target)
print(" Time Required for Random Forest in sec: " , (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
KNN(modifiedData, target)
print(" Time Required for KNN in sec: ", (time.time() - start_time))
print("----------------------------------------------------------------------")

start_time = time.time()
SVM(modifiedData, target)
print(" Time Required for SVM in sec: ", (time.time() - start_time))
print("----------------------------------------------------------------------")
plotGraph()


