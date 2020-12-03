
class Data:
    def __init__(self, name, logfile):
        self.name = name
        self.logfile = logfile

        self.train = None
        self.test = None

    def __str__(self):
        return "Data: %s" % self.name

    def prepare(self, setting):
        self.logfile.k = setting.prefixsize

        if setting.add_end:
            self.logfile.add_end_events()

        self.logfile.convert2int()
        self.logfile.create_k_context()

        self.train, self.test = self.logfile.splitTrainTest(setting.prefixsize, setting.split_cases, setting.split_data)