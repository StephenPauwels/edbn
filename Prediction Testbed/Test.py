from setting import CAMARGO
from data import get_data
from method import CAMARGO as m_camargo
from metric import ACCURACY

if __name__ == "__main__":
    d = get_data("BPIC12W")
    m = m_camargo
    s = CAMARGO
    e = ACCURACY

    s.prefixsize = 10

    # Standard steps for training and testing
    d.prepare(s)
    m.train(d)
    acc = m.test(d, e)
    print(acc)

