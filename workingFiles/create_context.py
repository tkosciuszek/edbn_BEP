from Data.data import Data
from Utils.LogFile import LogFile
import pandas as pd
import Predictions.setting as setting
import PrefixTreeCDDmain.settings as settings
data = pd.read_csv("Data/BPIC15_1_sorted_new.csv")

data = data[["case", "event", "role", "completeTime"]]

settings.init()
s = setting.STANDARD
s.train_percentage = 100

d = Data('BPIC_2015', LogFile(data, time_attr="completeTime", trace_attr="case",
                                  activity_attr='event', convert=False))
d.prepare(s)
saveFile = d.train.contextdata
saveFile.to_csv(path_or_buf="Data/BPIC15_1_context.csv")