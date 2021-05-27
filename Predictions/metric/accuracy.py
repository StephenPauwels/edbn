
class Accuracy:
    def __init__(self):
        pass

    def calculate(self, result):
        sum = 0
        total = 0
        for r in result:
            if r[0] == r[1]:
               sum += 1
            total += 1
        return sum / total