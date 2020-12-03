from Utils.LogFile import LogFile
from data.data import Data

HELPDESK = Data("Helpdesk", LogFile("../Data/Helpdesk.csv", ",", 0, None, "completeTime", "case", activity_attr="event", convert=False))
HELPDESK.logfile.keep_attributes(["event", "case", "completeTime"])
