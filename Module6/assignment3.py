import pandas as pd
import numpy as np

# Load data and drop unwanted column
X = pd.read_csv('D:/DAT210x-master/Module6/Datasets/parkinsons.data')
X.drop(labels=['name'], axis=1, inplace=True)

# Splice out columns as label
y = X['status']
X.drop(labels=['status'], axis=1, inplace=True)

# Prepare training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Do preprocessing
from sklearn import preprocessing
#normalizer = preprocessing.Normalizer().fit(X_train) 
#normalizer = preprocessing.MaxAbsScaler().fit(X_train)
#normalizer = preprocessing.MinMaxScaler().fit(X_train)
#normalizer = preprocessing.KernelCenterer().fit(X_train)
normalizer = preprocessing.StandardScaler().fit(X_train)
X_train = normalizer.transform(X_train)
X_test = normalizer.transform(X_test)

# Perform PCA or Isomap

#from sklearn.decomposition import PCA
#model = PCA(n_components=14).fit(X_train)
from sklearn import manifold
model = manifold.Isomap(n_neighbors=5, n_components=6).fit(X_train)
X_train = model.transform(X_train)
X_test = model.transform(X_test)


# Use for-loop to find best combination od c and gamma parameter for SVC in achieving highest record
best_score = 0
best_C = 0
best_gamma = 0
for C in np.arange(0.05, 2, 0.05) :
    for gamma in np.arange(0.001, 0.1, 0.001):
        # Create classifier svc
        from sklearn.svm import SVC
        svc = SVC(kernel='rbf', C=C, gamma=gamma)
       
        # Training svc and Score it to test data
        svc.fit(X_train, y_train)
        score = svc.score(X_test, y_test)
        if score > best_score :
            best_score = score
            best_C = C
            best_gamma = gamma

print best_score, best_C, best_gamma