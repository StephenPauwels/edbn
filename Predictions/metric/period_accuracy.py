import matplotlib.pyplot as plt

class Period_Accuracy:
    def __init__(self, num_events):
        self.events = num_events

    def calculate(self, result):
        accuracies = []
        result = list(result)
        for i in range(0, len(result), self.events):
            sum = 0
            total = 0
            for r in result[i:i+self.events]:
                if r[0] == r[1]:
                    sum += 1
                total += 1
            accuracies.append(sum / total)
        return accuracies
