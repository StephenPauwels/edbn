from Methods.EDBN.ConditionalTable import ConditionalTable

from keras.layers import Input, Embedding, Concatenate, Dense, Softmax, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Nadam
import keras.utils as ku
from keras.callbacks import EarlyStopping

import numpy as np


class NNT(ConditionalTable):
    def __init__(self, attr_name):
        super().__init__(attr_name)
        self.num_values = 0
        self.model = None

    def get_values(self, parent_val):
        inputs = []
        for i in range(len(self.parents)):
            if parent_val[i] not in self.parents[i].values:
                inputs.append(np.array([max(self.parents[i].values) + 1]))
            else:
                inputs.append(np.array([parent_val[i]]))

        prediction = self.model.predict(inputs)[0]
        best = np.argmax(prediction)

        return {best: prediction[best]}

    def check_parent_combination(self, parent_combination):
        return True

    def get_parent_combinations(self):
        return []

    def train(self, log):
        if len(self.parents) < 2:
            return

        model = self.construct_network(log)
        inputs = []
        for _ in range(len(self.parents)):
            inputs.append([])
        outputs = []
        for row in log.iterrows():
            for i in range(len(self.parents)):
                inputs[i].append(getattr(row[1], self.parents[i].attr_name))
            outputs.append(getattr(row[1], self.attr_name))

        outputs = ku.to_categorical(outputs, self.num_values + 1)
        for i in range(len(inputs)):
            inputs[i] = np.array(inputs[i])

        early_stopping = EarlyStopping(monitor='val_loss', patience=42)

        model.fit(x=inputs, y=outputs,
                  validation_split=0.2,
                  verbose=2,
                  callbacks=[early_stopping],
                  batch_size=32,
                  epochs=200)

        self.model = model

    def construct_network(self, log):
        # Input + Embedding layer for every parent

        input_layers = []
        embedding_layers = []
        for parent in self.parents:
            i = Input(shape=(1,), name=parent.attr_name.replace(" ", "_").replace("(", "").replace(")","").replace(":","_"))
            input_layers.append(i)
            e = Embedding(log[parent.attr_name].max() + 2, 32, embeddings_initializer="zeros")(i)
            embedding_layers.append(e)
        concat = Concatenate(name="concat")(embedding_layers)

        # dense1 = Dense(32)(concat)
        drop = Dropout(0.2)(concat)
        dense2 = Dense(log[self.attr_name].max() + 1)(drop)

        flat = Flatten()(dense2)

        output = Softmax(name="output")(flat)

        model = Model(inputs=input_layers, outputs=[output])
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        model.compile(loss={'output': 'categorical_crossentropy'},
                      optimizer=opt)
        model.summary()
        return model

    def test(self, row):
        return 1

