from Utils.LogFile import LogFile
from Predictions import edbn_adapter as edbn
from Predictions import base_adapter as baseline
from RelatedMethods.DiMauro import adapter as dimauro

if __name__ == "__main__":
    """
    data = "../Data/BPIC12.csv"
    logfile = LogFile(data, ",", 0, None, None, "case",
                      activity_attr="event", convert=False, k=2)

    logfile.keep_attributes(["case", "event", "role"])
    print("Convert")
    logfile.convert2int()
    print("Create k-context")
    logfile.create_k_context()

    print("PREPARING DONE")
    log_size = []
    acc = []
    for use_percentage in [0.1, 0.2, 0.5, 0.75]:
        new_log, _ = logfile.splitTrainTest(use_percentage, case=False, method="train-test")
        print("Used events:", new_log.contextdata.shape[0])
        log_size.append(new_log.contextdata.shape[0])
        train, test = new_log.splitTrainTest(70, case=False, method="test-train")
        acc.append(dimauro.test(test, dimauro.train(train, epochs=100, early_stop=10)))

    print(acc)
    print(log_size)
    """
    percent = [1, 2, 3, 4, 5, 10, 25, 50, 75, 100]
    log_size = [191, 382, 955, 1432, 1909, 3817, 5725, 7634, 9542, 19083, 47707, 95414, 143121, 190827]
    edbn_acc = [0.7931034482758621, 0.8173913043478261, 0.8362369337979094, 0.772093023255814, 0.8202443280977313, 0.7905759162303665, 0.8084982537834692, 0.8035792230467045, 0.8005588543485854, 0.8064628820960699, 0.7911688674631454, 0.7974497816593886, 0.801220392668328, 0.7977082569127845]
    base_acc = [0.8275862068965517, 0.808695652173913, 0.8083623693379791, 0.7767441860465116, 0.7888307155322862, 0.7827225130890052, 0.7951105937136205, 0.7961588825840245, 0.8050995459308418, 0.8064628820960699, 0.8094739048417523, 0.8055545851528384, 0.8052961315415609, 0.8084857377421439]
    dimauro_acc = [0.7758620689655172, 0.7913043478260869, 0.8118466898954704, 0.8046511627906977, 0.8167539267015707, 0.7923211169284468, 0.8026775320139697, 0.8048886948930598, 0.8026545581557807, 0.8031441048034934, 0.801578984140292, 0.7300960698689957, 0.7912755898176398, 0.794668902513581]

    import matplotlib.pyplot as plt
    plt.plot(log_size, edbn_acc, "-o", label="edbn")
    plt.plot(log_size, base_acc, "-o", label="base")
    plt.plot(log_size, dimauro_acc, "-o", label="di mauro")
    plt.show()
