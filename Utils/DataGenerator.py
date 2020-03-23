"""
    Author: Stephen Pauwels
"""

import copy
import random


class DataGenerator():
    def __init__(self):
        pass


class Variable():
    def __init__(self, num_vals, case_equal, prefix = ""):
        self.equal = case_equal
        self.num_vals = num_vals
        self.prefix = prefix

    def populate(self, sequence, anomaly=False):
        if self.equal:
            if self.num_vals > -1:
                choice = self.prefix + str(random.randint(0, self.num_vals))
            else:
                choice = self.prefix + str(round(random.random() * (10 ** random.randint(1,10))))
            for event in sequence:
                event.append(choice)

        else:
            for event in sequence:
                if self.num_vals > -1:
                    event.append(self.prefix + str(random.randint(0, self.num_vals)))
                else:
                    event.append(self.prefix + str(round(random.random() * (10 ** random.randint(1,10)))))


class Mapping(Variable):
    def __init__(self, unique, map_to, num_vals, prefix = ""):
        self.unique = unique
        self.map_to = map_to
        self.num_vals = num_vals
        self.mapping = {}
        self.values = set()
        self.prefix = prefix

    def populate(self, sequence, anomaly=False):
        for event in sequence:
            if anomaly:
                event.append(self.prefix + str("Anom"))
            else:
                if event[self.map_to] not in self.mapping:
                    if self.unique:
                        self.mapping[event[self.map_to]] = len(self.values)
                        self.values.add(len(self.values))
                    else:
                        self.mapping[event[self.map_to]] = random.randint(0, self.num_vals)

                event.append(self.prefix + str(self.mapping[event[self.map_to]]))


# TODO: add relation -> multiple possible values for map, each with a probability
class Relation(Variable):
    def __init__(self, map_to ):
        pass

class Sequence(Variable):
    def __init__(self, prefix = ""):
        self.sequences = []
        self.probs = []
        self.prefix = prefix

    def add_sequence(self, seq, prob, linked_sequence = None):
        sequence = []
        for val_idx in range(len(seq)):
            val_list = [self.prefix + seq[val_idx]]
            if linked_sequence:
                if isinstance(linked_sequence[0], list):
                    for link in linked_sequence:
                        val_list.append(link[val_idx])
                else:
                    val_list.append(linked_sequence[val_idx])
            sequence.append(val_list)
        self.sequences.append(sequence)

        if len(self.probs) == 0:
            self.probs.append(prob)
        else:
            self.probs.append(self.probs[-1] + prob)

    def create_case(self):
        choice = random.randint(0,100)
        for i in range(len(self.probs)):
            if choice <= self.probs[i]:
                return copy.deepcopy(self.sequences[i])


class DataModel():
    def __init__(self):
        self.sequence = None
        self.variables = []

    def setSequence(self, seq):
        self.sequence = seq

    def addVariable(self, var):
        self.variables.append(var)

    def generateCase(self, anomaly=False):
        seq = self.sequence.create_case()
        for v in self.variables:
            v.populate(seq, anomaly)
        return seq

def flatten_list(par_list):
    seqs = [[]]
    for val in par_list:
        if isinstance(val, list):
            new_seqs = []
            for s in seqs:
                for v in val:
                    new_seqs.append(s + [v])
            seqs = new_seqs

        else:
            for s in seqs:
                s.append(val)
    return seqs

def create_bohmer_synth_data(training_size, test_size, train_anoms, test_anoms, train_file, test_file, train_file_bohmer, test_file_bohmer):
    var1 = Mapping(False, 0, 50, prefix="r_")
    var2 = Mapping(False, 1, 7, prefix="wd_")

    # TODO: add a_, r_ and wd_ to the appropriate variables

    ###
    # Create correct model
    ###
    seq_var = Sequence(prefix="a_")

    activities = ["Package Goods", "Decide if normal post or special shipment", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 15)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 15)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 40)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 10)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 15)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 5)


    correct_model = DataModel()
    correct_model.setSequence(seq_var)
    correct_model.addVariable(var1)
    correct_model.addVariable(var2)

    ###
    # Create anomalous model (based on correct model)
    ###
    anomalous_model = DataModel()

    seq_var = Sequence(prefix="a_")
    activities = ["Decide if normal post or special shipment", "Package Goods", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 20)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 20)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 30)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    seq_var.add_sequence(activities, 30)

    anomalous_model.setSequence(seq_var)
    anomalous_model.addVariable(var1)
    anomalous_model.addVariable(var2)


    ###
    # Generate Data
    ###
    events_train_edbn = []
    events_test_edbn = []
    events_train_edbn.append(
        "pActivity,pResource,pWeekday,pCase,pAnomaly,"
        "Activity,Resource,Weekday,Case,Anomaly\n")
    events_test_edbn.append(events_train_edbn[0])

    events_train = []
    events_test = []
    events_train.append(
        "Activity,Resource,Weekday,Case,Anomaly\n")
    events_test.append(events_train[0])

    ###
    # Create training dataset (no anomalies)
    ###
    normals = 0
    anoms = 0
    for i in range(training_size):
        if random.randint(0,1000) < train_anoms:
            seq = anomalous_model.generateCase()
            anoms += 1
            # Output format for EDBN
            events_train_edbn.append(",".join(["START"] * len(seq[0])) + "," + str(i) + ",1," + ",".join(seq[0]) + "," + str(i) + ",1\n")
            for j in range(1, len(seq)):
                events_train_edbn.append(",".join(seq[j-1]) + "," + str(i) + ",1," + ",".join(seq[j]) + "," + str(i) + ",1\n")
            events_train_edbn.append(",".join(seq[-1]) + "," + str(i) + ",1," + ",".join(["END"] * len(seq[0])) + "," + str(i) + ",1\n")

            # Output format for Bohmer
            for s in seq:
                events_train.append(",".join(s) + "," + str(i) + ",1\n")
        else:
            seq = correct_model.generateCase()
            normals += 1
            # Output format for EDBN
            events_train_edbn.append(",".join(["START"] * len(seq[0])) + "," + str(i) + ",0," + ",".join(seq[0]) + "," + str(i) + ",0\n")
            for j in range(1, len(seq)):
                events_train_edbn.append(",".join(seq[j-1]) + "," + str(i) + ",0," + ",".join(seq[j]) + "," + str(i) + ",0\n")
            events_train_edbn.append(",".join(seq[-1]) + "," + str(i) + ",0," + ",".join(["END"] * len(seq[0])) + "," + str(i) + ",0\n")

            # Output format for Bohmer
            for s in seq:
                events_train.append(",".join(s) + "," + str(i) + ",0\n")
    print("Generated Training set with", normals, "normals and", anoms, "anomalies")

    ###
    # Create test dataset
    ###
    normals = 0
    anoms = 0
    added = False
    for i in range(test_size):
        if random.randint(0,1000) < test_anoms:
            anoms += 1
            if random.randint(0,100) < 80:
                seq = anomalous_model.generateCase()
            else:
                seq = correct_model.generateCase(True)

            # Output format for EDBN
            events_test_edbn.append(",".join(["START"] * len(seq[0])) + "," + str(i) + ",1," + ",".join(seq[0]) + "," + str(i) + ",1\n")
            for j in range(1, len(seq)):
                events_test_edbn.append(",".join(seq[j-1]) + "," + str(i) + ",1," + ",".join(seq[j]) + "," + str(i) + ",1\n")
            events_test_edbn.append(",".join(seq[-1]) + "," + str(i) + ",1," + ",".join(["END"] * len(seq[0])) + "," + str(i) + ",1\n")

            # Output format for Bohmer
            for s in seq:
                events_test.append(",".join(s) + "," + str(i) + ",1\n")
        else:
            normals += 1
            seq = correct_model.generateCase()

            # Output format for EDBN
            events_test_edbn.append(",".join(["START"] * len(seq[0])) + "," + str(i) + ",0," + ",".join(seq[0]) + "," + str(i) + ",0\n")
            for j in range(1, len(seq)):
                events_test_edbn.append(",".join(seq[j-1]) + "," + str(i) + ",0," + ",".join(seq[j]) + "," + str(i) + ",0\n")
            events_test_edbn.append(",".join(seq[-1]) + "," + str(i) + ",0," + ",".join(["END"] * len(seq[0])) + "," + str(i) + ",0\n")

            # Output format for Bohmer
            for s in seq:
                events_test.append(",".join(s) + "," + str(i) + ",0\n")
    print("Generated Testing set with", normals, "normals and", anoms, "anomalies")



    with open(train_file, "w") as fout:
        for e in events_train_edbn:
            fout.write(e)
    with open(test_file, "w") as fout:
        for e in events_test_edbn:
            fout.write(e)
    with open(train_file_bohmer, "w") as fout:
        for e in events_train:
            fout.write(e)
    with open(test_file_bohmer, "w") as fout:
        for e in events_test:
            fout.write(e)


def create_shipment_data(training_size, test_size, train_anoms, test_anoms, train_file, test_file):
    var1 = Variable(10, case_equal = True)
    var2 = Variable(-1, case_equal = False)
    var3 = Mapping(False, 4, 5)
    var4 = Mapping(True, 4, 1)
    var5 = Variable(100, case_equal = False)
    var6 = Mapping(False, 8, 18)
    var7 = Variable(25, case_equal = True)
    var8 = Mapping(True, 10, 5)
    var9 = Mapping(True, 10, 9)

    ###
    # Create correct model
    ###
    seq_var = Sequence()

    activities = ["Package Goods", "Decide if normal post or special shipment", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    label = [["Special", "Special", "Special", "Special", "Special"], ["<4000", "<4000", "<4000", "<4000", "<4000"], ["", "", "", "", ""]]
    seq_var.add_sequence(activities, 15, label)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    label = [["Special", "Special", "Special", "Special", "Special"], [">4000", ">4000", ">4000", ">4000", ">4000"], ["", "", "", "", ""]]
    seq_var.add_sequence(activities, 15, label)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000"], ["No", "No", "No", "No", "No"]]
    seq_var.add_sequence(activities, 40, label)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal"], [">4000", ">4000", ">4000", ">4000", ">4000"], ["No", "No", "No", "No", "No"]]
    seq_var.add_sequence(activities, 10, label)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"], [">4000", ">4000", ">4000", ">4000", ">4000", ">4000"], ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]]
    seq_var.add_sequence(activities, 15, label)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000", "<4000"], ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes"]]
    seq_var.add_sequence(activities, 5, label)


    correct_model = DataModel()
    correct_model.setSequence(seq_var)
    correct_model.addVariable(var1)
    correct_model.addVariable(var2)
    correct_model.addVariable(var3)
    correct_model.addVariable(var4)
    correct_model.addVariable(var5)
    correct_model.addVariable(var6)
    correct_model.addVariable(var7)
    correct_model.addVariable(var8)
    correct_model.addVariable(var9)

    ###
    # Create anomalous model (based on correct model)
    ###
    anomalous_model = DataModel()

    seq_var = Sequence()
    activities = ["Decide if normal post or special shipment", "Package Goods", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000"], ["", "", "", "", ""]]
    seq_var.add_sequence(activities, 20, label)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Request quotes from carriers", "Assign a carrier & prepare paperwork", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000"], ["", "", "", "", ""]]
    seq_var.add_sequence(activities, 20, label)

    activities = ["Decide if normal post or special shipment", "Package Goods", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000", "<4000"], ["No", "No", "No", "No", "No", "No"]]
    seq_var.add_sequence(activities, 30, label)

    activities = ["Package Goods", "Decide if normal post or special shipment", "Check if extra insurance is necessary", "Fill in a Post label", "Take out extra insurance", "Add paperwork and move package to pick area"]
    label = [["Normal", "Normal", "Normal", "Normal", "Normal", "Normal"], ["<4000", "<4000", "<4000", "<4000", "<4000", "<4000"], ["No", "No", "No", "No", "No", "No"]]
    seq_var.add_sequence(activities, 30, label)

    anomalous_model.setSequence(seq_var)
    anomalous_model.addVariable(var1)
    anomalous_model.addVariable(var2)
    anomalous_model.addVariable(var3)
    anomalous_model.addVariable(var4)
    anomalous_model.addVariable(var5)
    anomalous_model.addVariable(var6)
    anomalous_model.addVariable(var7)
    anomalous_model.addVariable(var8)
    anomalous_model.addVariable(var9)


    ###
    # Generate Data
    ###
    events_train = []
    events_test = []
    events_train.append(
        "Activity,Type,value,Insurance,Attr1,Attr2,Attr3,Attr4,Attr5,Attr6,Attr7,Attr8,Attr9,Case,Anomaly\n")
    events_test.append(events_train[0])

    ###
    # Create training dataset
    ###
    normals = 0
    anoms = 0
    for i in range(training_size):
        if random.randint(0,1000) < train_anoms:
            seq = anomalous_model.generateCase()
            anoms += 1
            for j in range(0, len(seq)):
                events_train.append(",".join(seq[j]) + "," + str(i) + ",1\n")
        else:
            seq = correct_model.generateCase()
            normals += 1
            for j in range(0, len(seq)):
                events_train.append(",".join(seq[j]) + "," + str(i) + ",0\n")
    print("Generated Training set with", normals, "normals and", anoms, "anomalies")

    ###
    # Create test dataset
    ###
    normals = 0
    anoms = 0
    added = False
    for i in range(test_size):
        if random.randint(0,1000) < test_anoms:
            anoms += 1
            if random.randint(0, 100) < 80:
                seq = correct_model.generateCase(True)
            else:
                seq = anomalous_model.generateCase()
            for j in range(0, len(seq)):
                events_test.append(",".join(seq[j]) + "," + str(i) + ",1\n")
        else:
            normals += 1
            seq = correct_model.generateCase()
            for j in range(0, len(seq)):
                events_test.append(",".join(seq[j]) + "," + str(i) + ",0\n")
    print("Generated Testing set with", normals, "normals and", anoms, "anomalies")



    with open(train_file, "w") as fout:
        for e in events_train:
            fout.write(e)
    with open(test_file, "w") as fout:
        for e in events_test:
            fout.write(e)


if __name__ == "__main__":
    #train_file = "Experiment_check3_train.csv"
    #test_file = "Experiment_check3_test.csv"

    #create_shipment_data(100000, 100000, 5, 10, train_file, test_file)

    #error_rates = [(0, 50), (0,25), (0,10), (0,5), (0,1), (5,50), (5, 25), (5, 10), (5,5), (5,1), (10,50), (10, 25), (10, 10), (10,5), (10,1)]
    #for i in range(len(error_rates)):
    #    train_file = "Experiment_train_" + str(i) + "_100000_" + str(error_rates[i][0]) + ".csv"
    #    test_file = "Experiment_test_" + str(i) + "_100000_" + str(error_rates[i][1]) + ".csv"
    #    create_shipment_data(100000, 100000, error_rates[i][0], error_rates[i][1], train_file, test_file)

    create_bohmer_synth_data(1000, 1000, 0, 10, "/home/spauwels/Data/Shipment-Bohmer/train_edbn.csv", "/home/spauwels/Data/Shipment-Bohmer/test_edbn.csv",
                             "/home/spauwels/Data/Shipment-Bohmer/train.csv", "/home/spauwels/Data/Shipment-Bohmer/test.csv")

    """
    path = "/Users/Stephen/git/interactive-log-mining/Datasets/Shipment - Bohmer/"
    error_rates = [(50, 100)]
    for i in range(len(error_rates)):
        train_file_bohmer = path + "Exp_%i_train_%i.csv" % (i, error_rates[i][0])
        train_file = path + "Exp_%i_train_%i_edbn.csv" % (i, error_rates[i][0])

        test_file_bohmer = path + "Exp_%i_test_%i.csv" % (i, error_rates[i][1])
        test_file = path + "Exp_%i_test_%i_edbn.csv" % (i, error_rates[i][1])
        create_bohmer_synth_data(10000, 10000, error_rates[i][0], error_rates[i][1], train_file, test_file, train_file_bohmer, test_file_bohmer)
    """