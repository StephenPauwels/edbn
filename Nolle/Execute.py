import arrow
import socket
from sqlalchemy.orm import Session

from Nolle.binet.april.fs import get_event_log_files, DATE_FORMAT
from Nolle.binet.april.anomalydetection import *
from Nolle.binet.april.database import EventLog, Model, get_engine
from Nolle.binet.april.dataset import Dataset
from Nolle.binet.april import Evaluator


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


if __name__ == "__main__":
    dataset = sorted([e.name for e in get_event_log_files() if e.p == 0.3 and e.name.startswith("small")])
    fit_and_save(dataset[3], BINetv3, fit_kwargs=dict(epochs=20, batch_size=500))
