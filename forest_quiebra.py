from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from sklearn.preprocessing import StandardScaler
f=open('Attr.txt','r')
lines = list(f)
df1=pd.DataFrame([])
for i in range(1,6):
    data = arff.loadarff(str(i)+'year.arff')
    df = pd.DataFrame(data[0])
    df1=df1.append(df)
df1=df1.dropna()
dfa=np.array(df1)
var=np.unique(dfa[:,-1])
dfa[:,-1]=dfa[:,-1]==var[1]
Target=dfa[:,-1]
Target=Target.astype('int')
Data=dfa[:,:-1]
x_train,x_1,y_train,y_1= sklearn.model_selection.train_test_split(Data, Target, train_size=0.5)
x_test,x_val,y_test,y_val= sklearn.model_selection.train_test_split(x_1, y_1, train_size=0.6)
n_trees = np.arange(1,10,1)
f1_train = []
f1_test = []
f1_val = []
feature_importance = np.zeros((len(n_trees),64))
for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test)))
    f1_val.append(sklearn.metrics.f1_score(y_val, clf.predict(x_val)))
    feature_importance[i, :] = clf.feature_importances_
maxtree=np.argmax(f1_test)+1
a = pd.Series(feature_importance[maxtree-1,:],lines)
a.nlargest().plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title(str(maxtree)+' arboles, F1-score=%4.3f'%f1_val[maxtree-1])
plt.savefig('features.png',bbox_inches='tight')