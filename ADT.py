import pandas as pd
import numpy as np
import hashlib
from collections.abc import Iterable
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import time

# define gvars
SEED = 1080
VCHUNK = 64

# import from csv to dataframe
df = pd.read_csv('wnl/WNL_TLS_Dataset_ECH.csv', sep='\t')

'''https://tls13.xargs.org/#client-hello/annotated'''

def flatten(L):
    '''flattens a list of nested list of arbitrary depth into a single concatenated list'''
    for x in L:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def padding(A, size):
    '''pads a given list to a given length'''
    t = size - len(A)

    return np.pad(A, pad_width=(0, t), mode='constant')

def parse_CHP(p):
    '''
    implements the Shamsimukhametov et al. bytes recomposition algorithm
    builds a fixed length feature vector from clienthello message
    '''
    T = []
    L = []
    V = []

    handshake_len = sum(p[3:5])
    clienthello_len = sum(p[6:9])
    sid_len = p[43]
    i = 44+sid_len

    sid = p[44:i]
    ciphersuite_len = sum(p[i:i+2])
    ciphersuite = p[i+2:i+2+ciphersuite_len]
    i += 4 + ciphersuite_len

    ext_len = sum(p[i:i+2])
    i = i+2
    
    T.append(padding(sid, 32))
    L.append(handshake_len)
    L.append(clienthello_len)
    L.append(sid_len)
    L.append(ciphersuite_len)
    L.append(ext_len)
    V.append(ciphersuite)
    
    end = i + ext_len
    count = 0   

    while (i < len(p)):

        n_ext_id = sum(p[i:i+2])
        i+=2
        
        n_ext_len = sum(p[i:i+2])
        i+=2

        T.append(n_ext_id)
        L.append(n_ext_len)
        V.append(p[i:i+n_ext_len])
        i+=n_ext_len

    T = padding(list(flatten(T)),40)
    L = padding(list(flatten(L)),13)
    V = list(flatten(V))
    VC = V[:VCHUNK]

    return np.concatenate((T,L,VC))

def parse_SHP(p):
    '''
    implements the Shamsimukhametov et al. bytes recomposition algorithm
    builds a fixed length feature vector from serverhello message
    '''
    T = []
    L = []
    V = []

    handshake_len = sum(p[3:5])
    serverhello_len = sum(p[6:9])
    sid_len = p[43]
    i = 44+sid_len

    sid = p[44:i]
    ciphersuite = sum(p[i:i+2])
    i += 3

    ext_len = sum(p[i:i+2])
    i = i+2
    
    T.append(padding(sid, 32))
    T.append(ciphersuite)
    L.append(handshake_len)
    L.append(serverhello_len)
    L.append(sid_len)
    L.append(ext_len)

    end = i + ext_len
    count = 0    
    
    while (i < len(p)):

        n_ext_id = sum(p[i:i+2])
        i+=2
        
        n_ext_len = sum(p[i:i+2])
        i+=2

        T.append(n_ext_id)
        L.append(n_ext_len)
        V.append(p[i:i+n_ext_len])
        i+=n_ext_len

    T = padding(list(flatten(T)),40)
    L = padding(list(flatten(L)),10)
    V = list(flatten(V))
    VC = V[:VCHUNK]

    return np.concatenate((T,L,VC))

def parse_payload(CH,SH):
    '''builds a concatenated fixed length feature vector of recomposed bytes'''
    chp  = np.fromstring(CH, dtype=int, sep=',')
    shp  = np.fromstring(SH, dtype=int, sep=',')
    chv = parse_CHP(chp)
    shv = parse_SHP(shp)
    return np.concatenate((chv,shv))

# Transform according to recomposition algorithm
df['X'] = df.apply(lambda row : parse_payload(row['ClientHello'], row['ServerHello']), axis=1)

# Encode classes into integers as terget variable vector y
labels = df.Label.unique()
df['Label'] = df['Label'].astype('category')
df['target'] = df['Label'].cat.codes
y = df['target'].to_numpy()

# expand array to proper dims
dfx = pd.DataFrame(df['X'].tolist()).add_prefix("x")
X = dfx.to_numpy(dtype=np.int16)

# split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED, shuffle=True, stratify=y)

# build model using optimized params
dtc_opt_model = DecisionTreeClassifier(   criterion='entropy', 
                                    max_depth=8, 
                                    max_features=None, 
                                    min_samples_leaf=5, 
                                    min_samples_split=2,                                     
                                    splitter='random',
                                    random_state=SEED)

ada_opt_model = AdaBoostClassifier( base_estimator=dtc_opt_model, 
                                    learning_rate=1.0, 
                                    n_estimators=190, 
                                    random_state=SEED)
# start timing
start = time.time()
# fit the ada boost
ada_opt_model.fit(X_train,y_train)
# stop timing
end = time.time()

# Make predictions on our test data
y_pred = ada_opt_model.predict(X_test)

# Evaluate the models performance (I hope this isnt garbage)
print(classification_report(y_test, y_pred, target_names=labels, digits=4))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training Time: "+str(end-start))