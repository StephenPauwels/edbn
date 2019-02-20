from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import sklearn as sk
from tqdm import tqdm

from april import Evaluator
from april.anomalydetection.utils import anomaly_ratio
from april.anomalydetection.utils import max_collapse
from april.enums import Axis
from april.fs import get_model_files
from april.utils import microsoft_colors

sns.set_style('white')

datasets = ['small']
binet_ads = ["BINETv3"]
ads = binet_ads

models = [m for m in get_model_files()]

frames = []
for model in tqdm(models):
    evaluator = Evaluator(model)

    binarizer = evaluator.binarizer
    targets = binarizer.get_targets(axis=2)
    scores = binarizer.mask(evaluator.result.scores)

    for i, attribute_name in enumerate(evaluator.dataset.attribute_keys):
        for axis in [0, 1, 2]:
            _targets = max_collapse(targets[:, :, i], axis=axis).compressed()
            _scores = max_collapse(scores[:, :, i], axis=axis).compressed()
            auc = sk.metrics.roc_auc_score(_targets, _scores)
            fpr, tpr, thresholds = sk.metrics.roc_curve(_targets, _scores)
            frames.append(pd.DataFrame(dict(ad=evaluator.ad_.name,
                                            axis=axis,
                                            dataset_id=model.id,
                                            process_model=evaluator.process_model_name,
                                            event_log_name=evaluator.eventlog_name,
                                            attribute_name=attribute_name,
                                            level='cf' if i == 0 else 'data',
                                            dataset_type='bpic' if 'bpic' in evaluator.eventlog_name else 'synthetic',
                                            tpr=tpr,
                                            fpr=fpr,
                                            auc=auc)))
df = pd.concat(frames)

dff = df.query('dataset_id == 4 and attribute_name == "name" and axis == 0')

g = plt.subplots(figsize=(10, 10))
g = sns.lineplot(data=dff, x='fpr', y='tpr')
plt.show()