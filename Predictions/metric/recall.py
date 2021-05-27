
class Recall:
    def __init__(self):
        pass

    def calculate(self, result):
        # all_predicted_vals = set([r[0] for r in result])

        correct_predicted = {}
        total_value = {}

        for r in result:
            expected_val = r[0]
            predicted_val = r[1]

            if expected_val not in total_value:
                total_value[expected_val] = 0
            total_value[expected_val] += 1

            if r[0] == r[1]:
                if predicted_val not in correct_predicted:
                   correct_predicted[predicted_val] = 0
                correct_predicted[predicted_val] += 1

        sum = 0
        for val in total_value.keys():
            sum += total_value[val] * (correct_predicted.get(val, 0) / total_value[val])
        return sum / len(result)

    