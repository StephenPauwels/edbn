from data import all_data, get_data
from method import ALL as ALL_METHODS
from setting import ALL as ALL_SETTINGS
from metric import ACCURACY
import time

def ranking_experiments(output_file):
    for d in all_data:
        event_data = get_data(d)
        for s in ALL_SETTINGS:
            event_data.prepare(s)

            with open(output_file, "a") as fout:
                fout.write("Data: " + event_data.name + "\n")
                fout.write(s.to_file_str())
                fout.write("Date: " + time.strftime("%d.%m.%y-%H.%M", time.localtime()) + "\n")
                fout.write("------------------------------------\n")

            for m in ALL_METHODS:
                m.train(event_data)
                acc = m.test(event_data, ACCURACY)
                with open(output_file, "a") as fout:
                    fout.write(m.name + ": " + str(acc) + "\n")
            with open(output_file, "a") as fout:
                fout.write("====================================\n\n")


if __name__ == "__main__":
    ranking_experiments("ranking_results.txt")
