# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Embedding
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.optimizers import Nadam, Adam, SGD, Adagrad


def training_model(vec, ac_weights, rl_weights, output_folder, args, epochs, early_stop):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    print('Build model...')
    print(args)
# =============================================================================
#     Input layer
# =============================================================================
    ac_input = Input(shape=(vec['prefixes']['x_ac_inp'].shape[1], ), name='ac_input')
    rl_input = Input(shape=(vec['prefixes']['x_rl_inp'].shape[1], ), name='rl_input')

# =============================================================================
#    Embedding layer for categorical attributes        
# =============================================================================
    ac_embedding = Embedding(ac_weights.shape[0],
                            ac_weights.shape[1],
                            weights=[ac_weights],
                            input_length=vec['prefixes']['x_ac_inp'].shape[1],
                            trainable=False, name='ac_embedding')(ac_input)

    rl_embedding = Embedding(rl_weights.shape[0],
                            rl_weights.shape[1],
                            weights=[rl_weights],
                            input_length=vec['prefixes']['x_rl_inp'].shape[1],
                            trainable=False, name='rl_embedding')(rl_input)
# =============================================================================
#    Layer 1
# =============================================================================
    l1_c1 = LSTM(args['l_size'],
                  kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  dropout=0.2,
                  implementation=args['imp'])(ac_embedding)
    
    l1_c2 = LSTM(args['l_size'],
                  kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  dropout=0.2,
                  implementation=args['imp'])(rl_embedding)


# =============================================================================
#    Batch Normalization Layer
# =============================================================================
    batch1 = BatchNormalization()(l1_c1)
    batch2 = BatchNormalization()(l1_c2)

# =============================================================================
# The layer specialized in prediction
# =============================================================================
    l2_c1 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch1)
 
#   The layer specialized in role prediction
    l2_c2 = LSTM(args['l_size'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=args['imp'])(batch2)

    
# =============================================================================
# Output Layer
# =============================================================================
    act_output = Dense(ac_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='act_output')(l2_c1)

    role_output = Dense(rl_weights.shape[0],
                       activation='softmax',
                       kernel_initializer='glorot_uniform',
                       name='role_output')(l2_c2)

    model = Model(inputs=[ac_input, rl_input], outputs=[act_output, role_output])

    if args['optim'] == 'Nadam':
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    elif args['optim'] == 'Adam':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                   epsilon=None, decay=0.0, amsgrad=False)
    elif args['optim'] == 'SGD':
        opt = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    elif args['optim'] == 'Adagrad':
        opt = Adagrad(lr=0.01, epsilon=None, decay=0.0)

    model.compile(loss={'act_output':'categorical_crossentropy', 'role_output':'categorical_crossentropy'}, optimizer=opt)
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)
#
#    # Output file
    output_file_path = os.path.join(output_folder,
                                    'model_rd_' + str(args['l_size']) +
                                    ' ' + args['optim'] +
                                    '_{epoch:03d}-{val_loss:.2f}.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')
    lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.5,
                                   patience=10,
                                   verbose=0,
                                   mode='auto',
                                   min_delta=0.0001,
                                   cooldown=0,
                                   min_lr=0)

    model.fit({'ac_input':vec['prefixes']['x_ac_inp'],
               'rl_input':vec['prefixes']['x_rl_inp']},
              {'act_output':vec['next_evt']['y_ac_inp'],
               'role_output':vec['next_evt']['y_rl_inp']},
              validation_split=0.2,
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=vec['prefixes']['x_ac_inp'].shape[1],
              epochs=epochs)
    return model