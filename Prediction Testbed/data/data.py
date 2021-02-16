
class Data:
    def __init__(self, name, logfile):
        self.name = name
        self.logfile = logfile

        self.train = None
        self.test = None
        self.test_orig = None

    def __str__(self):
        return "Data: %s" % self.name

    def prepare(self, setting):
        print("PREPARE")
        if setting.prefixsize:
            self.logfile.k = setting.prefixsize
        else:
            prefix_size = max(self.logfile.get_cases().size())
            if prefix_size > 40:
                self.logfile.k = 40
            else:
                self.logfile.k = prefix_size

        if setting.add_end:
            self.logfile.add_end_events()

        print("CONVERT")
        self.logfile.convert2int()
        print("K-CONTEXT")
        self.logfile.create_k_context()
        self.logfile.contextdata = self.logfile.contextdata.sort_values(by=[self.logfile.time]).reset_index()

        print("SPLIT TRAIN-TEST")
        self.train, self.test_orig = self.logfile.splitTrainTest(setting.train_percentage, setting.split_cases, setting.split_data)
        self.train.contextdata = self.train.contextdata.sort_values(by=[self.train.time]).reset_index()
        self.test_orig.contextdata = self.test_orig.contextdata.sort_values(by=[self.train.time]).reset_index()

    def create_batch(self, split="normal", timeformat=None):
        if split == "normal":
            self.test = {"full": {"data": self.test_orig}}
        elif split == "day":
            self.test = self.test_orig.split_days(timeformat)
        elif split == "week":
            self.test = self.test_orig.split_weeks(timeformat)
        elif split == "month":
            self.test = self.test_orig.split_months(timeformat)