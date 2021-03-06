# -*- coding:utf-8 -*-
"""
https://arxiv.org/ftp/arxiv/papers/1807/1807.02291.pdf 

The pre-trained GloVe word embeddings could be downloaded at: 
https://nlp.stanford.edu/projects/glove/ 

The Yelp 2013, 2014 and 2015 datasets are at: 
https://figshare.com/articles/Yelp_2013/6292142 
https://figshare.com/articles/Untitled_Item/6292253 
https://figshare.com/articles/Yelp_2015/6292334 
Yelp_P, Amazon_P and Amazon_F datasets are at: 
https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
Use tensorflow gpu 1.4.1 with keras 2.1.5 numpy=1.15 h5py=2.8.0, python 2.7
If your GPU is not compatible with CUDA 8 just try using keras 2.1.5, it solved my problem!
Use tensorflow gpu 1.10.0 with keras 2.2.4 numpy=1.15 h5py=2.8.0, python 3.6(gpu or not both ok)
"""
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense

# load data
# Using TensorFlow backend.
# [0.7554654092092673, 0.6680324381862092]
df = pd.read_csv("yelp_2013.csv")
# df.sample(5000)

Y = df.stars.values - 1
Y = to_categorical(Y, num_classes=5)
X = df.text.values

# set hyper parameters
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
NUM_FILTERS = 50
MAX_LEN = 512
Batch_size = 100
EPOCHS = 10

# shuffle the data
indices = np.arange(X.shape[0])
np.random.seed(2018)
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(len(X), len(Y))
# training set validation set and test set
nb_validation_samples_val = int((VALIDATION_SPLIT + TEST_SPLIT) * X.shape[0])
nb_validation_samples_test = int(TEST_SPLIT * X.shape[0])

x_train = X[:-nb_validation_samples_val]
y_train = Y[:-nb_validation_samples_val]
x_val = X[-nb_validation_samples_val:-nb_validation_samples_test]
y_val = Y[-nb_validation_samples_val:-nb_validation_samples_test]
x_test = X[-nb_validation_samples_test:]
y_test = Y[-nb_validation_samples_test:]

# use tokenizer to build vocab
tokenizer1 = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer1.fit_on_texts(df.text)
vocab = tokenizer1.word_index

x_train_word_ids = tokenizer1.texts_to_sequences(x_train)
x_test_word_ids = tokenizer1.texts_to_sequences(x_test)
x_val_word_ids = tokenizer1.texts_to_sequences(x_val)

# pad sequences into the same length
x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=MAX_LEN)
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=MAX_LEN)
x_val_padded_seqs = pad_sequences(x_val_word_ids, maxlen=MAX_LEN)

# slice sequences into many subsequence
x_train_padded_seqs_split = []
for i in range(x_train_padded_seqs.shape[0]):
    split1 = np.split(x_train_padded_seqs[i], 8)
    a = []
    for j in range(8):
        s = np.split(split1[j], 8)
        a.append(s)
    x_train_padded_seqs_split.append(a)

x_test_padded_seqs_split = []
for i in range(x_test_padded_seqs.shape[0]):
    split1 = np.split(x_test_padded_seqs[i], 8)
    a = []
    for j in range(8):
        s = np.split(split1[j], 8)
        a.append(s)
    x_test_padded_seqs_split.append(a)

x_val_padded_seqs_split = []
for i in range(x_val_padded_seqs.shape[0]):
    split1 = np.split(x_val_padded_seqs[i], 8)
    a = []
    for j in range(8):
        s = np.split(split1[j], 8)
        a.append(s)
    x_val_padded_seqs_split.append(a)

# print(x_train_padded_seqs_split[:3])
# print(x_test_padded_seqs_split[:3])
# print(x_val_padded_seqs_split[:3])

# load pre-trained GLove word embeddings
print("using Glove embeddings")
glove_path = "glove.6B.200d.txt"
embeddings_index = {}
with open(glove_path, "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
# print(embeddings_index[:1])
print("found {} word vectors".format(len(embeddings_index)))

# use pre-trained glove word embeddings to initialize the embedding layer
embedding_matrix = np.random.random((MAX_NUM_WORDS + 1, EMBEDDING_DIM))
for word, i in vocab.items():
    if i < MAX_NUM_WORDS:
        embedding_vector = embeddings_index.get(word)
        # words not found in embedding index will be random initialized.
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(MAX_NUM_WORDS + 1, EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_LEN / 64, trainable=True)

# build model
print("build model")
input1 = Input(shape=(int(MAX_LEN/64),), dtype="int32")
embed = embedding_layer(input1)
gru1 = GRU(NUM_FILTERS, recurrent_activation="sigmoid", activation=None, return_sequences=False)(embed)
Encoder1 = Model(input1, gru1)

input2 = Input(shape=(8, int(MAX_LEN/64),), dtype="int32")
embed2 = TimeDistributed(Encoder1)(input2)
gru2 = GRU(NUM_FILTERS, recurrent_activation="sigmoid", activation=None, return_sequences=False)(embed2)
Encoder2 = Model(input2, gru2)

input3 = Input(shape=(8, 8, int(MAX_LEN/64),), dtype="int32")
embed3 = TimeDistributed(Encoder2)(input3)
gru3 = GRU(NUM_FILTERS, recurrent_activation="sigmoid", activation=None, return_sequences=False)(embed3)
preds = Dense(5, activation="softmax")(gru3)
model = Model(input3, preds)

print(Encoder1.summary())
print(Encoder2.summary())
print(model.summary())

# use adam optimizer
from keras.optimizers import Adam

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])

# save the best model on validation set
from keras.callbacks import ModelCheckpoint

savebestmodel = "save_model/SRNN(8,2)_yelp2013.h5"
checkpoint = ModelCheckpoint(savebestmodel, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
callbacks = [checkpoint]

print(len(y_val), len(x_val_padded_seqs_split))
print(len(y_test), len(x_test_padded_seqs_split))
print(len(y_train), len(x_train_padded_seqs_split))

model.fit(x=np.array(x_train_padded_seqs_split), y=y_train,
          validation_data=(np.array(x_val_padded_seqs_split), y_val),
          epochs=EPOCHS,
          batch_size=Batch_size,
          callbacks=callbacks, verbose=1)

# use best model to evaluate on test set
from keras.models import load_model

best_model = load_model(savebestmodel)
print(best_model.evaluate(np.array(x_test_padded_seqs_split), y_test, batch_size=Batch_size))
