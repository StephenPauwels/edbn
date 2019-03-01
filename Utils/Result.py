import math

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
            self.attributes = event.attributes.keys()

    def get_attribute_score(self, attribute):
        total = 0
        for event in self.events:
            total += event.get_attribute_score(attribute)
        return total / len(self.events)

    def get_attribute_scores(self):
        scores = {}
        for attr in self.attributes:
            scores[attr] = self.get_attribute_score(attr)
        return scores

    def get_total_score(self):
        total = 0
        attr_scores = self.get_attribute_scores()
        for attr in attr_scores:
            total += attr_scores[attr]
        print("total", total, attr_scores)
        return total

    def get_nr_events(self):
        return len(self.events)

    def get_first_event_index(self):
        return self.events[0].id

class Event_result:

    def __init__(self, id = None):
        self.attributes = {}
        self.id = id

    def set_attribute_score(self, attribute, score):
        if score <= 0:
            self.attributes[attribute] = -5
        else:
            self.attributes[attribute] = math.log(score)

    def get_attribute_score(self, attribute):
        if attribute in self.attributes:
            return self.attributes[attribute]
        else:
            return None

    def get_total_score(self):
        total = 0
        for attr in self.attributes:
            total += self.attributes[attr]
        return total