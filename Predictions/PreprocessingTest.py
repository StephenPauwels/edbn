import pandas as pd

import Camargo.support_modules.role_discovery as rl

time_format = "%Y/%m/%d %H:%M:%S.%f"

log = pd.read_csv("../Data/BPIC12.csv", header=0, delimiter=",", dtype="str")
print(log)
_, resource_table = rl.role_discovery(log, False, 0.5)
print(resource_table)

# Role discovery
log_df_resources = pd.DataFrame.from_records(resource_table)
log_df_resources = log_df_resources.rename(index=str, columns={"resource": "user"})

# Dataframe creation
log_df = pd.DataFrame.from_records(log)
log_df = log_df.merge(log_df_resources, on='user', how='left')
log_df = log_df.reset_index(drop=True)
print(log_df)
