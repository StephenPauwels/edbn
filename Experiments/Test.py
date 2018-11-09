from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from LogFile import LogFile

log = LogFile("../Data/bpic2018.csv", ",", 0, 30000, "startTime", "case")
log.create_k_context()


# Add arc from event_Prev0 to event
print("Creating Kernels")
kernel_comb_old = stats.gaussian_kde(log.contextdata["event"])
kernel_parent_old = None
kernel_comb_new = stats.gaussian_kde([log.contextdata["event"], log.contextdata["event_Prev0"]])
kernel_parent_new = stats.gaussian_kde(log.contextdata["completeTime"])

print("Calculating scores")
S_comb_old = np.sum(np.log10(kernel_comb_old(log.contextdata["event"])))
S_comb_new = np.sum(np.log10(kernel_comb_new([log.contextdata["event"], log.contextdata["event_Prev0"]])))
S_parent_new = np.sum(np.log10(kernel_parent_new(log.contextdata["event_Prev0"])))

print(-S_comb_old + S_comb_new - S_parent_new)
