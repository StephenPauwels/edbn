from Utils import Uncertainty_Coefficient as uc, BayesianNet as bn
from eDBN.Constraint_Bayesian_Network import ConstraintBayesianNetwork

import pandas as pd


def generate_model(data, k, remove_attrs, trace_attr, label_attr, normal_label, previous_vals = False):
    cbn = ConstraintBayesianNetwork(len(data.columns), k, trace_attr, label_attr, normal_label)
    nodes = []

    i = 0
    for column in data:
        new_vals = uc.calculate_new_values_rate(data[column])
        nodes.append(column)
        if column not in remove_attrs and new_vals == 1:#(new_vals == 1 / len(data[column])) or new_vals == 1:
            remove_attrs.append(column)
        print(column, new_vals)
        i += 1

    for remove in remove_attrs:
        nodes.remove(remove)
    data = data[nodes]

    # Combine event in traces TODO: extend for any k
    traces = data.groupby([trace_attr])
    contextdata = {}
    # Get Attributes
    attributes = list(data.columns)
    attributes.remove(trace_attr)
    nodes.remove(trace_attr)
    for attribute in attributes:
        new_vals = uc.calculate_new_values_rate(data[attribute])
        contextdata[attribute] = []
        cbn.add_variable(0, attribute, new_vals)
        for i in range(k):
            contextdata["Prev%i_" % (i) + attribute] = []
            nodes.append("Prev%i_" % (i) + attribute)
            cbn.add_variable(0, "Prev%i_" % (i) + attribute, new_vals)
    for trace in traces:
        datatrace = trace[1]
        for event in range(len(datatrace)):
            for attr in range(len(attributes)):
                contextdata[attributes[attr]].append(datatrace.iloc[event, attr])
                for i in range(k):
                    if event - 1 - i < 0:
                        contextdata["Prev%i_" % (i) + attributes[attr]].append(0)
                    else:
                        contextdata["Prev%i_" % (i) + attributes[attr]].append(datatrace.iloc[event-1-i, attr])

    contextdataframe = pd.DataFrame(data=contextdata)

    # Calculate Mappings
    mappings = uc.calculate_mappings(contextdataframe, attributes, k, 0.98, previous_vals)
    double_mappings = []
    whitelist = []
    print("MAPPINGS:")
    for mapping in mappings:
        cbn.add_mapping(mapping[0], mapping[1])
        print(mapping[0], "=>", mapping[1])
        if (mapping[1], mapping[0]) in mappings and False:
            double_mappings.append(mapping)
        else:
            whitelist.append((mapping[0], mapping[1]))


    # Remove redundant mappings to improve Bay Net discovery performance
    while True:
        print("Closure:", double_mappings)
        _, closure = get_max_tranisitive_closure(double_mappings)
        print(closure)
        if len(closure) == 0:
            break
        for i in range(0, len(closure)):
            if i != 0:
                nodes.remove(closure[i][0])
            double_mappings.remove(closure[i])
            double_mappings.remove((closure[i][1], closure[i][0]))
    while len(double_mappings) > 0:
        for m in whitelist[:]:
            if m[0] == double_mappings[0][0]:
                whitelist.remove(m)
        nodes.remove(double_mappings[0][0])
        print("remove", double_mappings[0][0])
        double_mappings.remove((double_mappings[0][1], double_mappings[0][0]))
        double_mappings.remove(double_mappings[0])


    restrictions = None
    if previous_vals:
        restrictions = []
        for attr1 in attributes:
            for attr2 in attributes:
                if attr1 != attr2:
                    restrictions.append((attr2, attr1))
                for i in range(k):
                    restrictions.append(("Prev%i_" % (i) + attr2, attr1))

    # Calculate Bayesian Network
    bay_net = bn.BayesianNetwork(contextdataframe)
    net = bay_net.hill_climbing_pybn(nodes, restrictions=restrictions, whitelist=whitelist, metric="AIC")

    print("Done")

    relations = []
    for edge in net.edges():
        relations.append((edge[0], edge[1]))

    print("Relations:")
    for relation in relations:
        if relation not in mappings:
            cbn.add_parent(relation[0], relation[1])
            print(relation[0], "->", relation[1])

    return cbn

def get_max_tranisitive_closure(relations, closure = None, size = 0, prefix = ""):
    if not closure:
        closure = []

    max_size = 0
    max_closure = []

    if len(closure) > 0 and closure[0][0] == closure[-1][1]:
        return size, closure

    print(closure, relations)
    for r in relations:
        if len(closure) == 0 or (r[0] == closure[-1][1] and r not in closure and (r[1], r[0]) not in closure):
            max, found_closure = get_max_tranisitive_closure(relations, closure + [r], size + 1, prefix + "  ")
            if max > max_size:
                max_size = max
                max_closure = found_closure

    return max_size, max_closure

