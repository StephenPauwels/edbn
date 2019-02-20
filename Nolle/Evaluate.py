import itertools
from multiprocessing.pool import Pool

import pandas as pd
from sklearn import metrics
from sqlalchemy.orm import Session
from tqdm import tqdm

from april.anomalydetection import BINet
from april.anomalydetection.utils import label_collapse
from april.database import Evaluation
from april.database import Model
from april.database import get_engine
from april.enums import Base
from april.enums import Heuristic
from april.enums import Strategy
from april.evaluator import Evaluator
from april.fs import get_model_files
from april.fs import PLOT_DIR

heuristics = [h for h in Heuristic.keys() if h not in [Heuristic.DEFAULT, Heuristic.MANUAL, Heuristic.RATIO,
                                                       Heuristic.MEDIAN, Heuristic.MEAN]]
params = [(Base.SCORES, Heuristic.DEFAULT, Strategy.SINGLE), *itertools.product([Base.SCORES], heuristics, Strategy.keys())]

def _evaluate(params):
    e, base, heuristic, strategy = params

    session = Session(get_engine())
    model = session.query(Model).filter_by(file_name=e.model_file.name).first()
    session.close()

    # Generate evaluation frames
    y_pred = e.binarizer.binarize(base=base, heuristic=heuristic, strategy=strategy, go_backwards=False)
    y_true = e.binarizer.get_targets()

    evaluations = []
    for axis in [0, 1, 2]:
        for i, attribute_name in enumerate(e.dataset.attribute_keys):
            def get_evaluation(label, precision, recall, f1):
                return Evaluation(model_id=model.id, file_name=model.file_name,
                                  label=label, perspective=perspective, attribute_name=attribute_name,
                                  axis=axis, base=base, heuristic=heuristic, strategy=strategy,
                                  precision=precision, recall=recall, f1=f1)

            perspective = 'Control Flow' if i == 0 else 'Data'
            if i > 0 and not e.ad_.supports_attributes:
                evaluations.append(get_evaluation('Normal', 0.0, 0.0, 0.0))
                evaluations.append(get_evaluation('Anomaly', 0.0, 0.0, 0.0))
            else:
                yp = label_collapse(y_pred[:, :, i:i + 1], axis=axis).compressed()
                yt = label_collapse(y_true[:, :, i:i + 1], axis=axis).compressed()
                p, r, f, _ = metrics.precision_recall_fscore_support(yt, yp, labels=[0, 1])
                evaluations.append(get_evaluation('Normal', p[0], r[0], f[0]))
                evaluations.append(get_evaluation('Anomaly', p[1], r[1], f[1]))

    return evaluations

def evaluate(model_name):
    e = Evaluator(model_name)

    _params = []
    for base, heuristic, strategy in params:
        if e.dataset.num_attributes == 1 and strategy in [Strategy.ATTRIBUTE, Strategy.POSITION_ATTRIBUTE]:
            continue
        if isinstance(e.ad_, BINet) and e.ad_.version == 0:
            continue
        if heuristic is not None and heuristic not in e.ad_.supported_heuristics:
            continue
        if strategy is not None and strategy not in e.ad_.supported_strategies:
            continue
        if base is not None and base not in e.ad_.supported_bases:
            continue
        _params.append([e, base, heuristic, strategy])

    return [_e for p in _params for _e in _evaluate(p)]

if __name__ == "__main__":
    models = sorted([m.name for m in get_model_files() if m.p == 0.3])

    evaluations = []
    with Pool() as p:
        for e in tqdm(p.imap(evaluate, models), total=len(models), desc='Evaluate'):
            evaluations.append(e)

    print(evaluations)

    # Write to database
    session = Session(get_engine())
    for e in evaluations:
        session.bulk_save_objects(e)
        session.commit()
    session.close()