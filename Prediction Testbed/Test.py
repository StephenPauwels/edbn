from setting import STANDARD
from data import get_data
from method import get_method
from metric import CUMM_ACCURACY, ACCURACY

import matplotlib.pyplot as plt
import tensorflow as tf

from RelatedMethods.DiMauro.adapter import update

if __name__ == "__main__":
    d = get_data("BPIC15_1")
    m = get_method("DIMAURO")
    s = STANDARD
    e = CUMM_ACCURACY
    accuracy = ACCURACY

    s.train_percentage = 50

    # Standard steps for training and testing
    d.prepare(s)
    d.create_batch("month", "%Y-%m-%d %H:%M:%S")

    # m.train(d)
    # m.model.save("dimauro_model")

    m.model = tf.keras.models.load_model('dimauro_model')

    results = []
    for t in d.test:
        batch = d.test[t]["data"]
        results.extend(m.test_func(batch, m.model))
        update(batch, m.model, num_epochs=1)

    acc = e.calculate(results)

    plt.plot(range(len(acc)), acc)
    plt.show()
