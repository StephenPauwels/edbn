"""
    Author: Stephen Pauwels
"""

def train(log):
    import Methods.EDBN.model.GenerateModel as gm

    cbn = gm.generate_model(log, False)
    cbn.train(log)
    return cbn


def update(model, log):
    import Methods.EDBN.model.GenerateModel as gm

    cbn = gm.update_model(log, True, model)
    cbn.train(log)
    return cbn

