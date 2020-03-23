"""
    Author: Stephen Pauwels
"""

import datetime
import random

import pandas as pd


def generate_duration(timings, process, event):
    if random.randint(0,3) == 0:
        return datetime.timedelta(0)
    return datetime.timedelta(0.5 + abs(random.gauss(timings[process][event][0], timings[process][event][1])))

def generate_start_date():
    base = datetime.datetime.strptime("2019-01-01", "%Y-%m-%d")
    date = datetime.timedelta(random.uniform(0, 365))
    return base + date

def generate(training_size, test_size, train_anoms, test_anoms, train_file, test_file):
    # Process A -> B -> C | D -> E -> F

    process_timing = {}
    process_timing[0] = {}
    process_timing[0]["A"] = (1,0.2)
    process_timing[0]["B"] = (5,3)
    process_timing[0]["C"] = (2,2)
    process_timing[0]["E"] = (10,4)
    process_timing[0]["F"] = (3,1)

    process_timing[1] = {}
    process_timing[1]["A"] = (3,1)
    process_timing[1]["B"] = (10,10)
    process_timing[1]["D"] = (5,3)
    process_timing[1]["E"] = (10,4)
    process_timing[1]["F"] = (7,6)
    process_timing[1]["G"] = (8,4)

    events = ["A", "B", "C", "D", "E", "F"]

    processes = [["A", "B", "C", "E", "F"], ["A", "B", "D", "E", "F", "G"]]

    resources = [["R_1"], ["R_2", "R_3"]]

    events_data = []
    date_data = []
    trace_data = []
    anom = []
    changed = []
    types = []
    process_data = []
    resource_data = []
    random_data = []
    for trace in range(test_size):
        process_idx = random.randint(0,1)
        process = processes[process_idx]
        resource = random.choice(resources[process_idx])
        trace_anom = random.randint(0,1000)
        anomaly = False
        if trace_anom < train_anoms:
            anomaly = True
        event_anom = random.randint(1,len(process) - 1)
        anom_delta = None
        if anomaly:
            anom_delta = datetime.timedelta(1 + abs(random.gauss(20,10)))
        i = 0
        date = generate_start_date()
        for event in process:
            events_data.append(event)
            process_data.append(process_idx)
            resource_data.append(resource)
            random_data.append(random.randint(0,1000))
            date = date + generate_duration(process_timing, process_idx, event)
            changed.append("0")
            if anomaly and i == event_anom:
                date = date + anom_delta
                changed[-1] = "1"
            date_data.append(date.strftime("%Y-%m-%d %H:%M:%S"))
            trace_data.append(trace)
            if anomaly:
                anom.append("1")
                types.append("Alter Duration")
            else:
                anom.append("0")
                types.append("")
            i += 1

    df_dict = {"event" : events_data, "date" : date_data, "trace" : trace_data, "process": process_data, "resource": resource_data, "random": random_data, "anomaly": anom, "changed": changed, "anom_types": types}
    df = pd.DataFrame(df_dict)
    df.to_csv(train_file)

    events_data = []
    date_data = []
    trace_data = []
    anom = []
    changed = []
    types = []
    process_data = []
    resource_data = []
    random_data = []
    for trace in range(test_size):
        process_idx = random.randint(0,1)
        process = processes[process_idx]
        resource = random.choice(resources[process_idx])
        trace_anom = random.randint(0,1000)
        anomaly = False
        if trace_anom < test_anoms:
            anomaly = True
        event_anom = random.randint(1,len(process) - 1)
        anom_delta = None
        if anomaly:
            anom_delta = datetime.timedelta(1 + abs(random.gauss(20,10)))
        i = 0
        date = generate_start_date()
        for event in process:
            events_data.append(event)
            process_data.append(process_idx)
            resource_data.append(resource)
            random_data.append(random.randint(0,1000))
            date = date + generate_duration(process_timing, process_idx, event)
            changed.append("0")
            if anomaly and i == event_anom:
                date = date + anom_delta
                changed[-1] = "1"
            date_data.append(date.strftime("%Y-%m-%d %H:%M:%S"))
            trace_data.append(trace)
            if anomaly:
                anom.append("1")
                types.append("Alter Duration")
            else:
                anom.append("0")
                types.append("")
            i += 1

    df_dict = {"event" : events_data, "date" : date_data, "trace" : trace_data, "process": process_data, "resource": resource_data, "random": random_data, "anomaly": anom, "changed": changed, "anom_types": types}
    df = pd.DataFrame(df_dict)
    df.to_csv(test_file)


if __name__ == "__main__":
    generate()
