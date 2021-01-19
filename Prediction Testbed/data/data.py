
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

        self.logfile.convert2int()
        self.logfile.create_k_context()

        self.train, self.test_orig = self.logfile.splitTrainTest(setting.train_percentage, setting.split_cases, setting.split_data)

    def create_batch(self, split="normal", timeformat=None):
        if split == "normal":
            self.test = {"full": {"data": self.test_orig}}
        elif split == "day":
            self.test = self.test_orig.split_days(timeformat)
        elif split == "week":
            self.test = self.test_orig.split_weeks(timeformat)
        elif split == "month":
            self.test = self.test_orig.split_months(timeformat)