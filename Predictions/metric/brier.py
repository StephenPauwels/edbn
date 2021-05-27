
class Brier:
    def __init__(self):
        pass

    def calculate(self, result):
        diff = 0
        for r in result:
            diff += (1 - float(r[3]))
        return diff / len(result)