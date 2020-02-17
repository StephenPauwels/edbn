import numpy as np
from hyperopt import hp
from tensorflow.keras.utils import to_categorical

from DiMauro.utils import load_data

train_input = "../Camargo/output_files/data/helpdesk/train_log.csv"
(X_train, y_train,
 X_test, y_test,
 vocab_size,
 max_length,
 n_classes,
 prefix_sizes) = load_data(train_input)

# categorical output
y_train = to_categorical(y_train)

emb_size = (vocab_size + 1 ) // 2 # --> ceil(vocab_size/2)

n_iter = 20

space = {'input_length':max_length, 'vocab_size':vocab_size, 'n_classes':n_classes, 'model_type':model_type, 'embedding_size':emb_size,
         'n_modules':hp.choice('n_modules', [1,2,3]),
         'batch_size': hp.choice('batch_size', [9,10]),
         'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01))}

