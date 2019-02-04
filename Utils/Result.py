import math

class Result:

    def __init__(self):
        self.traces = []

    def add_trace(self, trace):
        self.traces.append(trace)


class Trace_result:

    def __init__(self, id, time = None):
        self.events = []
        self.id = id
        self.time = time

    def add_event(self, event):
        self.events.append(event)

    def get_attribute_score(self, attribute):
        total = 0
        for event in self.events:
            total += event.get_attribute_score(attribute)
        return total / len(self.events)

    def get_total_score(self):
        total = 0
        for event in self.events:
            total += event.get_total_score()
        return total

    def get_nr_events(self):
        return len(self.events)

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
            print(attribute)
            return None

    def get_total_score(self):
        total = 0
        for attr in self.attributes:
            total += self.attributes[attr]
        return total