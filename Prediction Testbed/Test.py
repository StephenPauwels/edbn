from setting import STANDARD
from data import HELPDESK
from method import SDL
from metric import ACCURACY

if __name__ == "__main__":
    HELPDESK.prepare(STANDARD)
    SDL.train(HELPDESK, {"epochs": 200, "early_stop": 10})
    acc = SDL.test(HELPDESK, ACCURACY)
    print(acc)