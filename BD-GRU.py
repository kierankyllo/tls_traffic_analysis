import pandas as pd
import numpy as np
from collections.abc import Iterable
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import time
from keras.utils import plot_model
import matplotlib.pyplot as plt

# define gvars
SEED = 1080
VCHUNK = 64
NUM_CLASSES = 12

BATCH_SIZE = 32
SPLIT = 0.2
EPOCHS = 50
LR = 0.001
METRIC = 'val_accuracy'
MIN_DELTA = 1e-4
PATIENCE = 100
THRESHOLD = 0.5

# Set random seed
tf.random.set_seed(SEED)

# import from csv to dataframe
df = pd.read_csv('wnl/WNL_TLS_Dataset_ECH.csv', sep='\t')
esni = pd.read_csv('wnl/WNL_TLS_Dataset_ESNI.csv', sep='\t')

# set checkpoint path and filename
check_name = "WNL_CLASS_GRU.chkp"
check_path = "checkpoint/" + check_name

# define our callbacks
callbacks = [

    # define checkpoint callback
    tf.keras.callbacks.ModelCheckpoint (
        check_path,
        monitor= 'val_accuracy',
        verbose= 1,
        save_best_only= True,
        save_weights_only= True,
        mode= 'auto',
        save_freq='epoch',
        options=None,
        initial_value_threshold=THRESHOLD,
    ),

    # # define early stopping callback
    # tf.keras.callbacks.EarlyStopping(
    #     monitor=METRIC,
    #     min_delta=MIN_DELTA,
    #     patience=PATIENCE,
    #     verbose=1
    # ),

]

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
X = X/256

# split the training data into train and val_test sets using categorical stratification
X_train, X_val_test, y_train , y_val_test = train_test_split(   X,
                                                                y,
                                                                stratify=y,
                                                                shuffle=True,
                                                                test_size=SPLIT,
                                                                random_state=SEED
                                                                )

# split the val_test data into val and test sets using categorical stratification
X_val, X_test, y_val , y_test = train_test_split(       X_val_test,
                                                        y_val_test,
                                                        test_size=0.5,
                                                        random_state=SEED
                                                        )
# get sizes
len_a = len(X_train)
len_b = len(X_train[5])

# reshape to fit the model
X_train = tf.reshape(X_train,(len_a,len_b,1))

# check for data leakage
print('Data Leak Check')
# need method to check for duplicate rows in data
print(len(X_train))
print(len(X_val))
print(len(X_test))
print(len(X_train), len(y_train), len(X_val_test), len(y_val_test), len(X_val), len(X_test), len(y_val), len(y_test)), print(len(X_train[5]))

# compose the model architecture function
def compose_model():
    in_layer = layers.Input(shape=(len_b,1))
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(in_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D()(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D()(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool1D()(x)
    
    # generic stacked dense softmax classifier
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(len_b, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    out_layer = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return Model(inputs=in_layer, outputs=out_layer)

# instantiate the model
model = compose_model()

# load previously trained weights
#status = model.load_weights(check_path).expect_partial()

# inspect model stack
print(model.summary())

# compile and fit the model to the data
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
              metrics=['accuracy'])

# save model plot
plot_model(model, to_file='BDGRU.png')

# start timer
start = time.time()

# fit the model
history = model.fit (   X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        verbose=1,
                        callbacks=callbacks,
                        )
# stop timer
end = time.time()

# Make predictions on our test data
y_pred = model.predict(X_test)
y_pred = np.rint(y_pred)
y_pred = np.argmax(y_pred, axis=1)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('BD-GRU Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('bdgru_acc.jpg')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('BD-GRU Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('bdgru_loss.jpg')
plt.show()

# Evaluate the models performance (I hope this isnt garbage)
print(classification_report(y_test, y_pred, target_names=labels, digits=4))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Training Time: "+str(end-start))