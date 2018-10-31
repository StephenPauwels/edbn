from Utils import Uncertainty_Coefficient as uc, BayesianNet as bn
from eDBN.extended_Dynamic_Bayesian_Network import extendedDynamicBayesianNetwork

import pandas as pd


def generate_model(data, k, remove_attrs, trace_attr, label_attr, normal_label, previous_vals = False):
    print("GENERATE: initialize")
    cbn = extendedDynamicBayesianNetwork(len(data.columns), k, trace_attr, label_attr, normal_label)
    nodes = []

    for column in data:
        if column not in remove_attrs:
            nodes.append(column)
    data = data[nodes]

    # Get all normal attributes and remove the trace attribute
    attributes = list(data.columns)
    attributes.remove(trace_attr)
    nodes.remove(trace_attr)

    # Create the k-context of the data
    print("GENERATE: build k-context")

    data = cbn.create_k_context(data)

    # Add previous-attributes to the model
    for attribute in attributes:
        new_vals = uc.calculate_new_values_rate(data[attribute])
        cbn.add_variable(attribute, new_vals)
        for i in range(k):
            nodes.append(attribute + "_Prev%i" % (i))
            cbn.add_variable(attribute + "_Prev%i" % (i), new_vals)

    print("GENERATE: calculate mappings")

    # Calculate Mappings
    mappings = uc.calculate_mappings(data, attributes, k, 0.99)
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

    print("GENERATE: removing redundant mappings")

    # Remove redundant mappings to improve Bay Net discovery performance
    while True:
        _, closure = get_max_tranisitive_closure(double_mappings)
        if len(closure) == 0:
            break
        keep_node = closure[0][0]
        for i in closure:
            if i[0] != keep_node and i[0] in nodes:
                nodes.remove(i[0])
            if i[1] != keep_node and i[1] in nodes:
                nodes.remove(i[1])
            mappings.remove(i)
            mappings.remove((i[1], i[0]))
            double_mappings.remove(i)
            double_mappings.remove((i[1], i[0]))
    while len(double_mappings) > 0:
        mapping = double_mappings[0]
        mappings.remove(mapping)
        double_mappings.remove(mapping)
        nodes.remove(mapping[1])
        mappings.remove((mapping[1], mapping[0]))
        double_mappings.remove((mapping[1], mapping[0]))

    rem_maps = []
    for mapping in mappings:
        if mapping[0] not in nodes or mapping[1] not in nodes:
            rem_maps.append(mapping)
    for r in rem_maps:
        mappings.remove(r)


    restrictions = []
    for attr1 in attributes:
        for attr2 in attributes:
            if attr1 != attr2:
                restrictions.append((attr2, attr1))
            for i in range(k):
                restrictions.append((attr2 + "_Prev%i" % (i), attr1))

    print("GENERATE: Learn Bayesian Network")

    # Calculate Bayesian Network
    bay_net = bn.BayesianNetwork(data)
    net = bay_net.hill_climbing_pybn(nodes, restrictions=restrictions, whitelist=whitelist, metric="AIC")

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

    for r in relations:
        if len(closure) == 0 or (r[0] == closure[-1][1] and r not in closure and (r[1], r[0]) not in closure):
            max, found_closure = get_max_tranisitive_closure(relations, closure + [r], size + 1, prefix + "  ")
            if max > max_size:
                max_size = max
                max_closure = found_closure

    return max_size, max_closure

