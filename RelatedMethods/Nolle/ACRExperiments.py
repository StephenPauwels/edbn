import itertools
import socket
from multiprocessing.pool import Pool

import april.processmining.log
import arrow
import pandas as pd
from april.anomalydetection import *
from april.anomalydetection.utils import label_collapse
from april.database import Evaluation
from april.database import EventLog
from april.database import Model
from april.database import get_engine
from april.database.utils import import_eventlogs, import_process_maps
from april.dataset import Dataset
from april.evaluator import Evaluator
from april.fs import DATE_FORMAT
from april.fs import EVENTLOG_DIR
from april.fs import EventLogFile
from april.fs import PLOT_DIR
from april.fs import get_event_log_files
from april.fs import get_model_files
from april.fs import get_process_model_files
from april.generation import CategoricalAttributeGenerator
from april.generation.anomaly import *
from april.generation.utils import generate_for_process_model
from april.processmining import ProcessMap
from april.utils import prettify_dataframe
from sklearn import metrics
from sqlalchemy.orm import Session
from tqdm import tqdm

#####
# Generate Data Files
#####

## Define Anomalies
anomalies = [
    SkipSequenceAnomaly(max_sequence_size=2),
    ReworkAnomaly(max_distance=5, max_sequence_size=3),
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    LateAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=2),
    AttributeAnomaly(max_events=3, max_attributes=2)
]

## Generate Datasets
process_models = [m for m in get_process_model_files() if 'testing' not in m and 'paper' not in m]
for process_model in tqdm(process_models, desc='Generate'):
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, num_attr=[1, 2, 3, 4], seed=1337)



## Add Anomalies
np.random.seed(0)  # This will ensure reproducibility
ps = [0.3]
event_log_paths = [e.path for e in get_event_log_files(EVENTLOG_DIR) if 'bpic' in e.name and e.p == 0.0]

combinations = list(itertools.product(event_log_paths, ps))
for event_log_path, p in tqdm(combinations, desc='Add anomalies'):
    event_log_file = EventLogFile(event_log_path)
    event_log = april.processmining.log.EventLog.from_json(event_log_path)

    anomalies = [
        SkipSequenceAnomaly(max_sequence_size=2),
        ReworkAnomaly(max_distance=5, max_sequence_size=3),
        EarlyAnomaly(max_distance=5, max_sequence_size=2),
        LateAnomaly(max_distance=5, max_sequence_size=2),
        InsertAnomaly(max_inserts=2),
    ]

    if event_log.num_event_attributes > 0:
        anomalies.append(AttributeAnomaly(max_events=3, max_attributes=min(2, event_log.num_activities)))

    for anomaly in anomalies:
        # This is necessary to initialize the likelihood graph correctly
        anomaly.activities = event_log.unique_activities
        anomaly.attributes = [CategoricalAttributeGenerator(name=name, values=values) for name, values in
                              event_log.unique_attribute_values.items() if name != 'name']

    for case in tqdm(event_log):
        if np.random.uniform(0, 1) <= p:
            anomaly = np.random.choice(anomalies)
            anomaly.apply_to_case(case)
        else:
            NoneAnomaly().apply_to_case(case)

    event_log.save_json(str(EVENTLOG_DIR / f'{event_log_file.model}-{p}-{event_log_file.id}.json.gz'))


#####
# Dataset Information
#####
logs = sorted([e.name for e in get_event_log_files() if e.p == 0.3])
columns = ['name', 'base_name', 'num_cases', 'num_events', 'num_activities',
           'num_attributes', 'attribute_keys', 'attribute_dims',
           'min_attribute_dim', 'max_attribute_dim',
           'min_case_len', 'max_case_len', 'mean_case_len']
df = []
for log in tqdm(logs):
    d = Dataset(log)
    dim_min = d.attribute_dims[1:].astype(int).min() if d.attribute_dims[1:].size else None
    dim_max = d.attribute_dims[1:].astype(int).max() if d.attribute_dims[1:].size else None
    df.append([log, log.split('-')[0], d.num_cases, d.num_events, d.attribute_dims[0].astype(int),
               d.num_attributes - 1, d.attribute_keys[1:], d.attribute_dims[1:].astype(int), dim_min, dim_max,
               d.case_lens.min(), d.case_lens.max(), d.case_lens.mean().round(2)])
event_logs = pd.DataFrame(df, columns=columns)

event_logs[['base_name', 'num_activities', 'num_cases', 'num_events', 'min_attribute_dim', 'max_attribute_dim']].groupby('base_name').agg(['count', 'min', 'max'])

# Process Model Information
maps = sorted([m for m in get_process_model_files()])
df = []
for process_map in tqdm(maps):
    model = ProcessMap.from_plg(process_map)

    num_variants = len(model.variants.cases)
    max_case_len = model.variants.max_case_len

    nodes = model.graph.number_of_nodes()
    edges = model.graph.number_of_edges()
    dens = nx.density(model.graph)
    in_degree = np.mean([d[1] for d in model.graph.in_degree()])
    out_degree = np.mean([d[1] for d in model.graph.out_degree()])

    df.append([nodes, edges, num_variants, max_case_len, dens, in_degree, out_degree])
process_models = pd.DataFrame(df, index=maps, columns=['nodes', 'edges', 'num_variants', 'max_case_len', 'density', 'in_deg', 'out_deg'])

event_logs.to_csv("datasets_events.csv")
process_models.to_csv("datasets_processes.csv")

import_process_maps()
import_eventlogs()


#####
# Train models
#####
def fit_and_save(dataset_name, ad, ad_kwargs=None, fit_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    # Save start time
    start_time = arrow.now()

    # Dataset
    dataset = Dataset(dataset_name)

    # AD
    ad = ad(**ad_kwargs)

    # Train and save
    ad.fit(dataset, **fit_kwargs)
    file_name = f'{dataset_name}_{ad.abbreviation}_{start_time.format(DATE_FORMAT)}'
    model_file = ad.save(file_name)

    # Save end time
    end_time = arrow.now()

    # Cache result
    Evaluator(model_file.str_path).cache_result()

    # Calculate training time in seconds
    training_time = (end_time - start_time).total_seconds()

    # Write to database
    engine = get_engine()
    session = Session(engine)

    session.add(Model(creation_date=end_time.datetime,
                      algorithm=ad.name,
                      training_duration=training_time,
                      file_name=model_file.file,
                      training_event_log_id=EventLog.get_id_by_name(dataset_name),
                      training_host=socket.gethostname(),
                      hyperparameters=str(dict(**ad_kwargs, **fit_kwargs))))
    session.commit()
    session.close()

    if isinstance(ad, NNAnomalyDetector):
        from keras.backend import clear_session
        clear_session()


datasets = sorted([e.name for e in get_event_log_files() if e.p == 0.3])

ads = [
    dict(ad=RandomAnomalyDetector),
    #dict(ad=BoehmerLikelihoodAnomalyDetector),
    dict(ad=DAE, fit_kwargs=dict(epochs=50, batch_size=500)),
    dict(ad=BINetv1, fit_kwargs=dict(epochs=20, batch_size=500)),
    dict(ad=BINetv2, fit_kwargs=dict(epochs=20, batch_size=500)),
    dict(ad=BINetv3, fit_kwargs=dict(epochs=20, batch_size=500))
]
for ad in ads:
    [fit_and_save(d, **ad) for d in tqdm(datasets, desc=ad['ad'].name)]


#####
# Evaluate models
#####

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

models = sorted([m.name for m in get_model_files()])

evaluations = []
with Pool() as p:
    for e in tqdm(p.imap(evaluate, models), total=len(models), desc='Evaluate'):
        evaluations.append(e)

# Write to database
session = Session(get_engine())
for e in evaluations:
    session.bulk_save_objects(e)
    session.commit()
session.close()

# Write to file
out_dir = PLOT_DIR / 'isj-2019'
eval_file = out_dir / 'eval.pkl'

session = Session(get_engine())
evaluations = session.query(Evaluation).all()
rows = []
for ev in tqdm(evaluations):
    m = ev.model
    el = ev.model.training_event_log
    rows.append([m.file_name, m.creation_date, m.hyperparameters, m.training_duration, m.training_host, m.algorithm,
                 el.file_name, el.base_name, el.percent_anomalies, el.number,
                 ev.axis, ev.base, ev.heuristic, ev.strategy, ev.label, ev.attribute_name, ev.perspective, ev.precision, ev.recall, ev.f1])
session.close()
columns = ['file_name', 'date', 'hyperparameters', 'training_duration', 'training_host', 'ad',
           'dataset_name', 'process_model', 'noise', 'dataset_id',
           'axis', 'base', 'heuristic', 'strategy', 'label', 'attribute_name', 'perspective', 'precision', 'recall', 'f1']
evaluation = pd.DataFrame(rows, columns=columns)
evaluation.to_pickle(eval_file)

#Output results
h_ads = ["DAE", "BINetv1", "BINetv2", "BINetv3"]
d_ads = ["Naive", "Sampling", "Likelihood", "OC-SVM"]

_filtered_evaluation = evaluation.query(f'ad in {h_ads} and (strategy == "{Strategy.ATTRIBUTE}"'
                                       f' or (strategy == "{Strategy.SINGLE}" and process_model == "bpic12")'
                                       f' or (strategy == "{Strategy.SINGLE}" and ad == "Naive+")) or ad in {d_ads}')

filtered_evaluation = _filtered_evaluation.query(f'heuristic == "{Heuristic.DEFAULT}"'
                                                 f' or (heuristic == "{Heuristic.LP_MEAN}" and ad != "DAE")'
                                                 f' or (heuristic == "{Heuristic.ELBOW_UP}" and ad == "DAE")')

df = filtered_evaluation.query('axis in [0, 2]')
df = prettify_dataframe(df)
df['f1'] = 2 * df['recall'] * df['precision'] / (df['recall'] + df['precision'])

df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model', 'dataset_name'], values=['precision', 'recall', 'f1'])
df = df.fillna(0)
df = df.stack(1).stack(1).reset_index()
df.to_csv("Results_detail")

df = pd.pivot_table(df, index=['axis', 'ad'], columns=['process_model'], values=['f1'], aggfunc=np.mean)
df = df.round(2)
df.to_csv("Results.csv")