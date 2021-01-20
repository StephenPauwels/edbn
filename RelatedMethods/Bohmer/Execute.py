"""
    Author: Stephen Pauwels
"""

from RelatedMethods.Bohmer import LikelihoodGraph as lg


def train(data, act_idx=1, res_idx=2, wk_idx=3):

    model = lg.LikelihoodModel(data, act_idx, res_idx, wk_idx)

    model.basicLikelihoodGraph()
    model.extendLikelihoodGraph()
    return model

def test(test, output_file, model, label, normal_val):

    test_cases = list(test.data.groupby(test.trace))

    total_tests = len(test_cases)
    i = 0
    scores = []
    for name, case in test_cases:
        print("Testing", i, "/", total_tests)
        scores.append((name, model.test_trace(case), case.iloc[0][label] != normal_val))
        i += 1

    scores.sort(key=lambda l: l[1])
    with open(output_file, "w") as fout:
        for s in scores:
            fout.write(",".join([str(i) for i in s]))
            fout.write("\n")
