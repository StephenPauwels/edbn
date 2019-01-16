from Utils import Uncertainty_Coefficient as uc, BayesianNet as bn
from eDBN.extended_Dynamic_Bayesian_Network import extendedDynamicBayesianNetwork

import pandas as pd


def generate_model(data, remove_attrs = []):
    print("GENERATE: initialize")
    # Initialize empty eDBN datastructure
    cbn = extendedDynamicBayesianNetwork(len(data.attributes()), data.k, data.trace)
    nodes = []

    # Remove attributes
    for column in data.attributes():
        if column not in remove_attrs:
            nodes.append(column)
    data.keep_attributes(nodes)

    # Get all normal attributes and remove the trace attribute
    attributes = list(data.attributes())
    attributes.remove(data.trace)
    nodes.remove(data.trace)

    if data.time in attributes:
        attributes.remove(data.time)
        nodes.remove(data.time)


    # Create the k-context of the data
    print("GENERATE: build k-context")

    data.create_k_context()

    # Add previous-attributes to the model
    for attribute in attributes:
        new_vals = uc.calculate_new_values_rate(data.get_column(attribute))
        print(attribute, new_vals)
        cbn.add_variable(attribute, new_vals)
        for i in range(data.k):
            nodes.append(attribute + "_Prev%i" % (i))
            cbn.add_variable(attribute + "_Prev%i" % (i), new_vals)

    print("GENERATE: calculate mappings")

    # Calculate Mappings
    mappings = uc.calculate_mappings(data.contextdata, attributes, data.k, 0.99)

    tmp_mappings = mappings[:]
    print("GENERATE: removing redundant mappings")
    ignore_nodes = []
    while True:
        cycle = get_max_closure(tmp_mappings)
        if len(cycle) == 0:
            break
        print("GENERATE: found cycle:", cycle)
        cycle_nodes = [c[0] for c in cycle]
        ignore_nodes.extend(cycle_nodes[1:])
        remove_mappings = []
        for m in tmp_mappings:
            if m[0] in cycle_nodes:
                remove_mappings.append(m)
        for rem in remove_mappings:
            tmp_mappings.remove(rem)

    for n in ignore_nodes:
        nodes.remove(n)

    whitelist = []
    print("MAPPINGS:")
    for mapping in mappings:
        cbn.add_mapping(mapping[0], mapping[1])
        print(mapping[0], "=>", mapping[1])
        if (mapping[0], mapping[1]) in tmp_mappings:
            whitelist.append((mapping[0], mapping[1]))

    # Create list with allowed edges (only from previous -> current and current -> current)
    restrictions = []
    for attr1 in attributes:
        for attr2 in attributes:
            if attr1 != attr2:
                restrictions.append((attr2, attr1))
            for i in range(data.k):
                restrictions.append((attr2 + "_Prev%i" % (i), attr1))

    print("GENERATE: Learn Bayesian Network")

    # Calculate Bayesian Network
    bay_net = bn.BayesianNetwork(data.contextdata)
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
    print(prefix, relations, closure, size)
    if not closure:
        closure = []

    max_size = 0
    max_closure = []

    if len(closure) > 0 and closure[0][0] == closure[-1][1]:
        return size, closure

    for r in relations:
        if len(closure) == 0 or (r[0] == closure[-1][1] and r not in closure):
            max, found_closure = get_max_tranisitive_closure(relations, closure + [r], size + 1, prefix + "  ")
            if max > max_size:
                max_size = max
                max_closure = found_closure

    return max_size, max_closure


def get_max_closure(relations, nodes = None, closure = None):
    if nodes is None:
        nodes = []

        for r in relations:
            if r[0] not in nodes:
                nodes.append(r[0])
            if r[1] not in nodes:
                nodes.append(r[1])

    if closure is None:
        closure = []

    if len(closure) > 0 and closure[0][0] == closure[-1][1]:
        return closure

    max_closure = []

    for r in relations:
        if len(closure) == 0 or (r[0] == closure[-1][1] and r[1] in nodes):
            found_closure = get_max_closure(relations, [n for n in nodes if n != r[1]], closure + [r])
            if len(found_closure) > len(max_closure):
                max_closure = found_closure
    return max_closure





if __name__ == "__main__":
    doubles = [('action_code', 'Activity'), ('concept_name', 'Activity'), ('Activity', 'action_code'), ('concept_name', 'action_code'), ('activityNameNL', 'activityNameEN'), ('activityNameEN', 'activityNameNL'), ('Activity', 'concept_name'), ('action_code', 'concept_name')]
    print(get_max_closure(doubles))
    doubles = [('activityNameNL', 'activityNameEN'), ('activityNameEN', 'activityNameNL')]
    print(get_max_closure(doubles))
