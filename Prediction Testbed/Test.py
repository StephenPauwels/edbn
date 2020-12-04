from setting import TAX
from data import get_data
from method import SDL
from metric import ACCURACY

if __name__ == "__main__":
    d = get_data("Helpdesk")
    m = SDL
    s = TAX
    e = ACCURACY

    # Standard steps for training and testing
    d.prepare(s)
    m.train(d)
    acc = m.test(d, e)
    print(acc)

