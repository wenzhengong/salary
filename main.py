import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from random import randrange, choice
from sklearn.preprocessing import scale, add_dummy_feature
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cross_validation import KFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC

# read datasets, return feature matrix and label vector
def read_data(filename):
    f = open(filename)
    tr_data = []
    tr_label = []
    for line in f:
        readline = line.split(', ')
        instance = []
        if not(any("?" in s for s in readline)):# do not pick up samples with missing values
            # create feature matrix
            instance.append(readline[0])
            # categorical feature: workclass
            if readline[1] == 'Private':
                temp = [0] * 8
                temp[0] = 1
                instance[1:9]= temp
            elif readline[1] == 'Self-emp-not-inc':
                temp = [0] * 8
                temp[1] = 1
                instance[1:9]= temp
            elif readline[1] == 'Self-emp-inc':
                temp = [0] * 8
                temp[2] = 1
                instance[1:9]= temp
            elif readline[1] == 'Federal-gov':
                temp = [0] * 8
                temp[3] = 1
                instance[1:9]= temp
            elif readline[1] == 'Local-gov':
                temp = [0] * 8
                temp[4] = 1
                instance[1:9]= temp
            elif readline[1] == 'State-gov':
                temp = [0] * 8
                temp[5] = 1
                instance[1:9]= temp
            elif readline[1] == 'Without-pay':
                temp = [0] * 8
                temp[6] = 1
                instance[1:9]= temp
            elif readline[1] == 'Never-worked':
                temp = [0] * 8
                temp[7] = 1
                instance[1:9]= temp
            instance.append(readline[2])
            # categorical feature: education
            if readline[3] == 'Bachelors':
                temp = [0] * 16
                temp[0] = 1
                instance[10:26]= temp
            elif readline[3] == 'Some-college':
                temp = [0] * 16
                temp[1] = 1
                instance[10:26]= temp
            elif readline[3] == '11th':
                temp = [0] * 16
                temp[2] = 1
                instance[10:26]= temp
            elif readline[3] == 'HS-grad':
                temp = [0] * 16
                temp[3] = 1
                instance[10:26]= temp
            elif readline[3] == 'Prof-school':
                temp = [0] * 16
                temp[4] = 1
                instance[10:26]= temp
            elif readline[3] == 'Assoc-acdm':
                temp = [0] * 16
                temp[5] = 1
                instance[10:26]= temp
            elif readline[3] == 'Assoc-voc':
                temp = [0] * 16
                temp[6] = 1
                instance[10:26]= temp
            elif readline[3] == '9th':
                temp = [0] * 16
                temp[7] = 1
                instance[10:26]= temp
            elif readline[3] == '7th-8th':
                temp = [0] * 16
                temp[8] = 1
                instance[10:26]= temp
            elif readline[3] == '12th':
                temp = [0] * 16
                temp[9] = 1
                instance[10:26]= temp
            elif readline[3] == 'Masters':
                temp = [0] * 16
                temp[10] = 1
                instance[10:26]= temp
            elif readline[3] == '1st-4th':
                temp = [0] * 16
                temp[11] = 1
                instance[10:26]= temp
            elif readline[3] == '10th':
                temp = [0] * 16
                temp[12] = 1
                instance[10:26]= temp
            elif readline[3] == 'Doctorate':
                temp = [0] * 16
                temp[13] = 1
                instance[10:26]= temp
            elif readline[3] == '5th-6th':
                temp = [0] * 16
                temp[14] = 1
                instance[10:26]= temp
            elif readline[3] == 'Preschool':
                temp = [0] * 16
                temp[15] = 1
                instance[10:26]= temp
            instance.append(readline[4])
            # categorical feature: marital-status
            if readline[5] == 'Married-civ-spouse':
                temp = [0] *7
                temp[0] = 1
                instance[27:34]= temp
            elif readline[5] == 'Divorced':
                temp = [0] *7
                temp[1] = 1
                instance[27:34]= temp
            elif readline[5] == 'Never-married':
                temp = [0] *7
                temp[2] = 1
                instance[27:34]= temp
            elif readline[5] == 'Separated':
                temp = [0] *7
                temp[3] = 1
                instance[27:34]= temp
            elif readline[5] == 'Widowed':
                temp = [0] *7
                temp[4] = 1
                instance[27:34]= temp
            elif readline[5] == 'Married-spouse-absent':
                temp = [0] *7
                temp[5] = 1
                instance[27:34]= temp
            elif readline[5] == 'Married-AF-spouse':
                temp = [0] *7
                temp[6] = 1
                instance[27:34]= temp
            # categorical feature: occupation
            if readline[6] == 'Tech-support':
                temp = [0] *14
                temp[0] = 1
                instance[34:48]= temp
            elif readline[6] == 'Craft-repair':
                temp = [0] *14
                temp[1] = 1
                instance[34:48]= temp
            elif readline[6] == 'Other-service':
                temp = [0] *14
                temp[2] = 1
                instance[34:48]= temp
            elif readline[6] == 'Sales':
                temp = [0] *14
                temp[3] = 1
                instance[34:48]= temp
            elif readline[6] == 'Exec-managerial':
                temp = [0] *14
                temp[4] = 1
                instance[34:48]= temp
            elif readline[6] == 'Prof-specialty':
                temp = [0] *14
                temp[5] = 1
                instance[34:48]= temp
            elif readline[6] == 'Handlers-cleaners':
                temp = [0] *14
                temp[6] = 1
                instance[34:48]= temp
            elif readline[6] == 'Machine-op-inspct':
                temp = [0] *14
                temp[7] = 1
                instance[34:48]= temp
            elif readline[6] == 'Adm-clerical':
                temp = [0] *14
                temp[8] = 1
                instance[34:48]= temp
            elif readline[6] == 'Farming-fishing':
                temp = [0] *14
                temp[9] = 1
                instance[34:48]= temp
            elif readline[6] == 'Transport-moving':
                temp = [0] *14
                temp[10] = 1
                instance[34:48]= temp
            elif readline[6] == 'Priv-house-serv':
                temp = [0] *14
                temp[11] = 1
                instance[34:48]= temp
            elif readline[6] == 'Protective-serv':
                temp = [0] *14
                temp[12] = 1
                instance[34:48]= temp
            elif readline[6] == 'Armed-Forces':
                temp = [0] *14
                temp[13] = 1
                instance[34:48]= temp
            # categorical feature: relationship
            if readline[7] == 'Wife':
                temp = [0] * 6
                temp[0] = 1
                instance[48:54]= temp
            elif readline[7] == 'Own-child':
                temp = [0] * 6
                temp[1] = 1
                instance[48:54]= temp
            elif readline[7] == 'Husband':
                temp = [0] * 6
                temp[2] = 1
                instance[48:54]= temp
            elif readline[7] == 'Not-in-family':
                temp = [0] * 6
                temp[3] = 1
                instance[48:54]= temp
            elif readline[7] == 'Other-relative':
                temp = [0] * 6
                temp[4] = 1
                instance[48:54]= temp
            elif readline[7] == 'Unmarried':
                temp = [0] * 6
                temp[5] = 1
                instance[48:54]= temp
            # categorical feature: race
            if readline[8] == 'White':
                temp = [0] * 5
                temp[0] = 1
                instance[54:59]= temp
            elif readline[8] == 'Asian-Pac-Islander':
                temp = [0] * 5
                temp[1] = 1
                instance[54:59]= temp
            elif readline[8] == 'Amer-Indian-Eskimo':
                temp = [0] * 5
                temp[2] = 1
                instance[54:59]= temp
            elif readline[8] == 'Other':
                temp = [0] * 5
                temp[3] = 1
                instance[54:59]= temp
            elif readline[8] == 'Black':
                temp = [0] * 5
                temp[4] = 1
                instance[54:59]= temp
            # categorical feature: sex
            if readline[9] == 'Female':
                temp = [0] * 2
                temp[0] = 1
                instance[59:61]= temp
            elif readline[9] == 'Male':
                temp = [0] * 2
                temp[1] = 1
                instance[59:61]= temp
            instance.append(readline[10])
            instance.append(readline[11])
            instance.append(readline[12])
            # categorical feature: native-country
            if readline[13] == 'United-States':
                temp = [0] * 41
                temp[0] = 1
                instance[64:105]= temp
            elif readline[13] == 'Cambodia':
                temp = [0] * 41
                temp[1] = 1
                instance[64:105]= temp
            elif readline[13] == 'England':
                temp = [0] * 41
                temp[2] = 1
                instance[64:105]= temp
            elif readline[13] == 'Puerto-Rico':
                temp = [0] * 41
                temp[3] = 1
                instance[64:105]= temp
            elif readline[13] == 'Canada':
                temp = [0] * 41
                temp[4] = 1
                instance[64:105]= temp
            elif readline[13] == 'Germany':
                temp = [0] * 41
                temp[5] = 1
                instance[64:105]= temp
            elif readline[13] == 'Outlying-US(Guam-USVI-etc)':
                temp = [0] * 41
                temp[6] = 1
                instance[64:105]= temp
            elif readline[13] == 'India':
                temp = [0] * 41
                temp[7] = 1
                instance[64:105]= temp
            elif readline[13] == 'Japan':
                temp = [0] * 41
                temp[8] = 1
                instance[64:105]= temp
            elif readline[13] == 'Greece':
                temp = [0] * 41
                temp[9] = 1
                instance[64:105]= temp
            elif readline[13] == 'South':
                temp = [0] * 41
                temp[10] = 1
                instance[64:105]= temp
            elif readline[13] == 'China':
                temp = [0] * 41
                temp[11] = 1
                instance[64:105]= temp
            elif readline[13] == 'Cuba':
                temp = [0] * 41
                temp[12] = 1
                instance[64:105]= temp
            elif readline[13] == 'Iran':
                temp = [0] * 41
                temp[13] = 1
                instance[64:105]= temp
            elif readline[13] == 'Honduras':
                temp = [0] * 41
                temp[14] = 1
                instance[64:105]= temp
            elif readline[13] == 'Philippines':
                temp = [0] * 41
                temp[15] = 1
                instance[64:105]= temp
            elif readline[13] == 'Italy':
                temp = [0] * 41
                temp[16] = 1
                instance[64:105]= temp
            elif readline[13] == 'Poland':
                temp = [0] * 41
                temp[17] = 1
                instance[64:105]= temp
            elif readline[13] == 'Jamaica':
                temp = [0] * 41
                temp[18] = 1
                instance[64:105]= temp
            elif readline[13] == 'Vietnam':
                temp = [0] * 41
                temp[19] = 1
                instance[64:105]= temp
            elif readline[13] == 'Mexico':
                temp = [0] * 41
                temp[20] = 1
                instance[64:105]= temp
            elif readline[13] == 'Portugal':
                temp = [0] * 41
                temp[21] = 1
                instance[64:105]= temp
            elif readline[13] == 'Ireland':
                temp = [0] * 41
                temp[22] = 1
                instance[64:105]= temp
            elif readline[13] == 'France':
                temp = [0] * 41
                temp[23] = 1
                instance[64:105]= temp
            elif readline[13] == 'Dominican-Republic':
                temp = [0] * 41
                temp[24] = 1
                instance[64:105]= temp
            elif readline[13] == 'Laos':
                temp = [0] * 41
                temp[25] = 1
                instance[64:105]= temp
            elif readline[13] == 'Ecuador':
                temp = [0] * 41
                temp[26] = 1
                instance[64:105]= temp
            elif readline[13] == 'Taiwan':
                temp = [0] * 41
                temp[27] = 1
                instance[64:105]= temp
            elif readline[13] == 'Haiti':
                temp = [0] * 41
                temp[28] = 1
                instance[64:105]= temp
            elif readline[13] == 'Columbia':
                temp = [0] * 41
                temp[29] = 1
                instance[64:105]= temp
            elif readline[13] == 'Hungary':
                temp = [0] * 41
                temp[30] = 1
                instance[64:105]= temp
            elif readline[13] == 'Guatemala':
                temp = [0] * 41
                temp[31] = 1
                instance[64:105]= temp
            elif readline[13] == 'Nicaragua':
                temp = [0] * 41
                temp[32] = 1
                instance[64:105]= temp
            elif readline[13] == 'Scotland':
                temp = [0] * 41
                temp[33] = 1
                instance[64:105]= temp
            elif readline[13] == 'Thailand':
                temp = [0] * 41
                temp[34] = 1
                instance[64:105]= temp
            elif readline[13] == 'Yugoslavia':
                temp = [0] * 41
                temp[35] = 1
                instance[64:105]= temp
            elif readline[13] == 'El-Salvador':
                temp = [0] * 41
                temp[36] = 1
                instance[64:105]= temp
            elif readline[13] == 'Trinadad&Tobago':
                temp = [0] * 41
                temp[37] = 1
                instance[64:105]= temp
            elif readline[13] == 'Peru':
                temp = [0] * 41
                temp[38] = 1
                instance[64:105]= temp
            elif readline[13] == 'Hong':
                temp = [0] * 41
                temp[39] = 1
                instance[64:105]= temp
            elif readline[13] == 'Holand-Netherlands':
                temp = [0] * 41
                temp[40] = 1
                instance[64:105]= temp
            tr_data.append(instance)
            # create label vector
            if readline[14] == '<=50K\n':
                tr_label.append('0')
            elif readline[14] == '>50K\n':
                tr_label.append('1')
    return (np.array(tr_data).astype(float), np.array(tr_label).astype(float))
# data standardization
def standardize(X):
    # use built-in function
    return scale(X)
# confusion matrix
def conf_mat(y_true, y_pred):
    cm1 = confusion_matrix(y_true, y_pred)
    df1 = pandas.DataFrame(cm1)
    tot0 = cm1[0, 0] + cm1[0, 1]
    tot1 = cm1[1, 0] + cm1[1, 1]
    cm2 = np.array([[format(cm1[0, 0]/tot0,'.2%'), format(cm1[0, 1]/tot0,'.2%')],[format(cm1[1, 0]/tot1,'.2%'), format(cm1[1, 1]/tot1,'.2%')]])
    df2 = pandas.DataFrame(cm2)
    print(df1)
    print(df2)
    correctrate = (cm1[0, 0] + cm1[1, 1])/(tot0 + tot1)
    print("The rate of accuracy is ", format(correctrate, '.2%'))
# returns training matrix after PCA
def PCA_r(X, dim):
    # can change n_components to desired value
    pca = PCA(n_components = dim)
    return pca.fit_transform(X)
# returns training matrix after FLD, however, dimsionality is restricted as 1, not interesting
def FLD_r(X, y):
    # can change n_components to desired value
    fld = LDA()
    return fld.fit_transform(X, y)
# get training and test sets of k fold cross validation
def get_kfold_train_test(X, y, k):
    kf = KFold(len(X), k)
    for train, test in kf:
        # training sets and test sets
        yield X[train], y[train], X[test], y[test]
# random assignment without prior
def RA(X_tr, y_tr, X_te):
    clf = DummyClassifier(strategy='uniform').fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred
# random assignment with prior 1: assign with P(S_i)
def RA1(X_tr, y_tr, X_te):
    clf = DummyClassifier(strategy='stratified').fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred
# random assignment with prior 2: assign with the maximum prior
def RA2(X_tr, y_tr, X_te):
    clf = DummyClassifier(strategy='most_frequent').fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred
# baseline system: minimum distance to class mean
def baseline(X_tr, y_tr, X_te):
    y_pred = []
    m0 = X_tr[np.where(y_tr == 0)].mean(0)
    m1 = X_tr[np.where(y_tr == 1)].mean(0)
    for idx in range(len(X_te)):
        if np.linalg.norm(X_te[idx] - m0) < np.linalg.norm(X_te[idx] - m1):
            y_pred.append(0)
        else:
            y_pred.append(1)
    return(np.array(y_pred).astype(float))
# pseudoinverse learning
def pinv(X_tr, y_tr, X_te):
    # augment the feature space
    X_tr_aug = add_dummy_feature(X_tr)
    X_te_aug = add_dummy_feature(X_te)
    X_tr_aug[np.where(y_tr == 1)] = -X_tr_aug[np.where(y_tr == 1)]
    b = np.ones((len(X_tr_aug),))
    w = np.dot(np.linalg.pinv(X_tr_aug), b)
    indicator = np.dot(X_te_aug, w)
    for i in range(len(indicator)):
        if indicator[i] > 0:
            indicator[i] = 0
        else:
            indicator[i] = 1
    return indicator
# perceptron
def percep(X_tr, y_tr, X_te):
    clf = Perceptron(n_iter = 1000)
    X_tr_aug = add_dummy_feature(X_tr)
    X_te_aug = add_dummy_feature(X_te)
    clf.fit(X_tr_aug, y_tr)
    y_pred = clf.predict(X_te_aug)
    return y_pred
# k nearest neighbors
def knn(X_tr, y_tr, X_te):
    neigh = KNeighborsClassifier(n_neighbors = int(math.sqrt(len(X_tr))))
    neigh.fit(X_tr, y_tr)
    y_pred = neigh.predict(X_te)
    return y_pred
# parzen window
def par(X_tr, y_tr, X_te, r):
    neigh = RadiusNeighborsClassifier(radius = r)
    neigh.fit(X_tr, y_tr)
    y_pred = neigh.predict(X_te)
    return y_pred
# SVM, takes more than an hour, takes forever when working with SMOTE algorithm
def svm(X_tr, y_tr, X_te):
    clf = SVC(kernel = 'linear')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_pred
# SMOTE algorithm
def SMOTE(T, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : array, shape = [(N/100) * n_minority_samples, n_features]
    """    
    n_minority_samples, n_features = T.shape
    
    if N < 100:
        #create synthetic samples only for a subset of T.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = int(N/100)
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    #Calculate synthetic samples
    for i in range(n_minority_samples):
        nn = neigh.kneighbors(T[i], return_distance=False)
        for n in range(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T[nn_index] - T[i]
            gap = np.random.random()
            S[n + i * N, :] = T[i,:] + gap * dif[:]
    
    return S
# plot prototypes
def plot(X_tr, y_tr):
    X_tr0 = X_tr[np.where(y_tr == 0)]
    X_tr1 = X_tr[np.where(y_tr == 1)]
    plt.plot(X_tr0[:,0], X_tr0[:,1], 'r.')
    plt.plot(X_tr1[:,0], X_tr1[:,1], 'g.')
    plt.show()

# read datasets
input('Press any key to load adults dataset(both training and test)...')
X_tr, y_tr = read_data('adult-data')
X_te, y_te = read_data('adult-test')
while(True):
    yorn = input('Do you want to apply SMOTE algorithm? Enter y or n: ')
    if yorn == 'y':
        # populate the minority class by SMOTE algorithm
        X_tr0 = X_tr[np.where(y_tr == 0)]
        X_tr1 = SMOTE(X_tr[np.where(y_tr == 1)], 300, 2)
        X_tr = np.concatenate((X_tr0, X_tr1))
        y_tr0 = y_tr[np.where(y_tr == 0)]
        y_tr1 = y_tr[np.where(y_tr == 1)]
        y_tr = np.concatenate((y_tr0, y_tr1, y_tr1, y_tr1))
        break
    elif yorn == 'n':
        break
    else:
        print("Please enter y or n only")
while(True):
    yorn1 = input('Do you want to standardize data? Enter y or n: ')
    if yorn1 == 'y':
        # data standardization, uncomment if you want to
        X_tr = standardize(X_tr)
        X_te = standardize(X_te)
        break
    elif yorn1 == 'n':
        break
    else:
        print("Please enter y or n only")
while(True):
    yorn = input('Do you want to use PCA? Enter y or n: ')
    if yorn == 'y':
        dim = input('How many dimensions do you want to reduce to? Enter an positive integer less than 105: ')
        PCA_r(X_tr, int(dim))
        PCA_r(X_te, int(dim))
        break
    elif yorn == 'n':
        break
    else:
        print("Please enter y or n only")
input('Press any key to launch RANDOM ASSIGNMENT WITHOUT PRIOR...')
conf_mat(y_te, RA(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch RANDOM ASSIGNMENT WITH PRIOR...')
conf_mat(y_te, RA1(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch RANDOM ASSIGNMENT WITH MAXIMUM PRIOR...')
conf_mat(y_te, RA2(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch MINIMUM TO CLASS MEAN CLASSIFIER (my baseline)...')
conf_mat(y_te, baseline(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch PSEUDOINVERSE LEARNING...')
conf_mat(y_te, pinv(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch PERCEPTRON LEARNING...')
conf_mat(y_te, percep(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch K NEAREST NEIGHBORS...')
conf_mat(y_te, knn(X_tr, y_tr, X_te))
print('=============================================')
input('Press any key to launch PARZEN WINDOW...')
if yorn1 == 'n':
    conf_mat(y_te, par(X_tr, y_tr, X_te, 40000))
else:
    conf_mat(y_te, par(X_tr, y_tr, X_te, 27))
print('=============================================')
while(True):
    yorn = input('Do you want to launch SUPPORT VECTOR MACHINE? Takes more than one hour, will not stop if using SMOTE. Enter y or n: ')
    if yorn == 'y':
        conf_mat(y_te, svm(X_tr, y_tr, X_te))
        break
    elif yorn == 'n':
        break
    else:
        print("Please enter y or n only")

