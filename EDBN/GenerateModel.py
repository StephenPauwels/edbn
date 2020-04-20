"""
    Author: Stephen Pauwels
"""

from EDBN.ExtendedDynamicBayesianNetwork import ExtendedDynamicBayesianNetwork
from EDBN.LearnBayesianStructure import Structure_learner
from Utils import Uncertainty_Coefficient as uc


def generate_model(data, remove_attrs = None):
    """
    Generate an EDBN model given the data

    :param data:            The data to learn the model from
    :param remove_attrs:    Attributes to be ignored
    :return:                The learned EDBN
    """
    # Initialize empty EDBN datastructure
    print("GENERATE: initialize")
    cbn = ExtendedDynamicBayesianNetwork(len(data.attributes()), data.k, data.trace)
    nodes = []


    # Remove attributes in remove_attrs
    if remove_attrs is None:
        remove_attrs = []

    for column in data.attributes():
        if column not in remove_attrs:
            nodes.append(column)
    data.keep_attributes(nodes)


    # Get all normal attributes and remove the trace and time attribute
    attributes = list(data.attributes())

    if data.trace in attributes:
        attributes.remove(data.trace)
        nodes.remove(data.trace)

    if data.time in attributes:
        attributes.remove(data.time)
        nodes.remove(data.time)


    # Create the k-context of the data
    print("GENERATE: build k-context")
    if data.trace is not None:
        data.create_k_context()


    # Add previous-attributes to the model
    for attribute in attributes:
        if data.isCategoricalAttribute(attribute):
            new_vals = uc.calculate_new_values_rate(data.get_column(attribute))
            empty_val = data.convert_string2int(attribute, "nan")
            cbn.add_discrete_variable(attribute, new_vals, empty_val)
            for i in range(data.k):
                nodes.append(attribute + "_Prev%i" % (i))
                cbn.add_discrete_variable(attribute + "_Prev%i" % (i), new_vals, empty_val)
        else:
            cbn.add_numerical_variable(attribute)
            for i in range(data.k):
                nodes.append(attribute + "_Prev%i" % (i))
                cbn.add_numerical_variable(attribute + "_Prev%i" % (i))


    # Add duration to the model
    for i in range(data.k):
        dur_attr = "duration_%i" % (i)
        if data.contextdata is not None and dur_attr in data.contextdata.columns:
            if data.isNumericAttribute(dur_attr):
               cbn.add_numerical_variable("duration_%i" % (i))
            else:
                cbn.add_discrete_variable("duration_0",uc.calculate_new_values_rate(data.contextdata["duration_0"]), data.convert_string2int(dur_attr, "nan"))
            nodes.append("duration_0")

    print("GENERATE: calculate mappings")

    # Calculate Mappings
    #mappings = uc.calculate_mappings(data, attributes, 0.99)
    mappings = []

    # Look for cycles in mappings
    tmp_mappings = mappings[:]
    print("GENERATE: removing redundant mappings")
    ignore_nodes = []
    while True:
        cycle = get_max_cycle(tmp_mappings)
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
#        nodes.remove(n)
        print("Remove", n)

    whitelist = []
    print("GENERATE: Found Functional Dependencies:")
    for mapping in mappings:
        cbn.get_variable(mapping[1]).add_mapping(cbn.get_variable(mapping[0]))
        print("   ", mapping[0], "=>", mapping[1])
        if (mapping[0], mapping[1]) in tmp_mappings and mapping[0] not in ignore_nodes and mapping[1] not in ignore_nodes:
            whitelist.append((mapping[0], mapping[1]))

    # Create list with allowed edges (only from previous -> current and current -> current)
    restrictions = []
    for attr1 in attributes:
        if attr1 != data.activity:
            continue
        for attr2 in attributes:
            for i in range(data.k):
                restrictions.append((attr2 + "_Prev%i" % (i), attr1))
        if "duration_0" in nodes:
            restrictions.append((attr1, "duration_0"))


    print("GENERATE: Learn Bayesian Network")

    # Calculate Bayesian Network
    learner = Structure_learner(data, nodes)
    # relations = learner.learn(restrictions)
    relations = learner.learn(restrictions, whitelist)

    print("GENERATE: Found Conditional Dependencies:")
    for relation in relations:
        if relation not in mappings:
            cbn.get_variable(relation[1]).add_parent(cbn.get_variable(relation[0]))
            print("   ", relation[0], "->", relation[1])

    return cbn


def get_max_cycle(relations, nodes = None, closure = None):
    """
    Given a list of tuples, return the maximum cycle found in the list

    :param relations:   a list of tuples, where every tuple denotes an edge in a graph
    :param nodes:       all nodes not yet visited
    :param closure:     current maximum cycle
    :return:            if a cycle exists: return the maximum cycle otherwise return empty list
    """
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
            found_closure = get_max_cycle(relations, [n for n in nodes if n != r[1]], closure + [r])
            if len(found_closure) > len(max_closure):
                max_closure = found_closure
    return max_closure
