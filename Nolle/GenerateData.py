from tqdm import tqdm

from april.fs import get_process_model_files
from april.generation.anomaly import *
from april.generation.utils import generate_for_process_model

anomalies = [
    SkipSequenceAnomaly(max_sequence_size=2),
    ReworkAnomaly(max_distance=5, max_sequence_size=3),
    EarlyAnomaly(max_distance=5, max_sequence_size=2),
    LateAnomaly(max_distance=5, max_sequence_size=2),
    InsertAnomaly(max_inserts=2),
    AttributeAnomaly(max_events=3, max_attributes=2)
]

process_models = [m for m in get_process_model_files() if 'testing' not in m and 'paper' not in m]
print(process_models)
for process_model in tqdm(process_models, desc='Generate'):
    generate_for_process_model(process_model, size=5000, anomalies=anomalies, num_attr=[1, 2, 3, 4], seed=1337)