import metric
import matplotlib.pyplot as plt

def read_file(filename):
    results = []
    with open(filename) as finn:
        for line in finn:
            split = line.split(",")
            results.append((int(split[0]), int(split[1]), float(split[2])))
    return results


if __name__ == "__main__":
    c_a = metric.CUMM_ACCURACY
    a = metric.ACCURACY

    results = read_file("results/DBN_BPIC15_1_day.csv")
    c_a_results = c_a.calculate(results)
    a_result = a.calculate(results)

    plt.plot(range(len(c_a_results)), c_a_results, label="update-retain (day)")
    plt.show()
    print("Accuracy:", a_result)