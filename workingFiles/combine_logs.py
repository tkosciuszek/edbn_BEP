import pandas as pd




df1 = pd.read_csv("Data/BPIC15_1_sorted_new.csv")
df2 = pd.read_csv("Data/BPIC15_2_sorted_new.csv")
df3 = pd.read_csv("Data/BPIC15_3_sorted_new.csv")
df4 = pd.read_csv("Data/BPIC15_4_sorted_new.csv")
df5 = pd.read_csv("Data/BPIC15_5_sorted_new.csv")

finaldf = pd.concat([df1, df2, df3, df4, df5])

finaldf.to_csv("Data/BPIC15_ALL.csv")
