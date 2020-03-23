"""
    Author: Stephen Pauwels
"""

class Result:

    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)


class Trace_result:

    def __init__(self, id, anomaly = False, time = None):
        self.events = []
        self.id = id
        self.time = time
        self.attributes = None
        self.anomaly = anomaly

    def __repr__(self):
        return "Trace_Result(" + str(self.id) + ")"

    def add_event(self, event):
        self.events.append(event)
        if self.attributes is None:
            self.attributes = []
            for score_pair in event.attributes:
                self.attributes.append(score_pair[0])

    def get_attribute_score(self, attribute):
        total = 0
        for event in self.events:
            total += event.get_attribute_score(attribute)
        return total / len(self.events)


    def get_attribute_score_per_event(self, attribute):
        scores = []
        for event in self.events:
            scores.append(event.get_attribute_score(attribute))
        return scores


    def get_attribute_scores(self, calibration = None):
        scores = {}
        if calibration is None:
            for attr in self.attributes:
                scores[attr] = self.get_attribute_score(attr)
        else:
            for attr in self.attributes:
                scores[attr] = self.get_attribute_score_calibrated(attr, calibration)
        return scores

    def get_total_score(self):
        total = 0
        attr_scores = self.get_attribute_scores()
        for attr in attr_scores:
            total += attr_scores[attr]
        return total

    def get_calibrated_score(self, calibration):
        total = 0
        attr_scores = self.get_attribute_scores(calibration)
        for attr in attr_scores:
            total += attr_scores[attr]
        return total

    def get_attribute_score_calibrated(self, attribute, calibration):
        total = 0
        for event in self.events:
            total += event.get_attribute_score(attribute)
        total /= len(self.events)

        #if total < calibration[attribute][0]:
        #    total = math.log(0.01)
        #else:
        #    total = math.log(np.searchsorted(calibration[attribute], total) / len(calibration[attribute]))

        return total# * calibration[attribute]


    def get_nr_events(self):
        return len(self.events)

    def get_first_event_index(self):
        return self.events[0].id

    def get_anom_type(self):
        return self.events[0].type

class Event_result:

    def __init__(self, id = None, type = None):
        self.attributes = []
        self.id = id
        self.type = type

    def set_attribute_score(self, attribute, score):
        self.attributes.append((attribute,score))

    def get_attribute_score(self, attribute):
        for score_pair in self.attributes:
            if score_pair[0] == attribute:
                return score_pair[1]
        return None

    def get_total_score(self):
        total = 0
        for score_pair in self.attributes:
            total += score_pair[1]
        return total