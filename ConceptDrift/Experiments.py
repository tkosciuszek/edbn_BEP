import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import ConceptDrift as cd
from Utils.LogFile import LogFile
#def __init__(self, filename, delim, header, rows, time_attr, trace_attr, activity_attr=None, values=None,
                 #integer_input=False, convert=True, k=1, dtype=None):
base_folder = '/Users/teddy/Documents/TU_e/ThirdYear/Quartile_4/BEP/edbn_BEP/'
timeattr = 'completeTime'
traceattr = 'case'
activityattr = 'event'
def learn_and_dump_model(dataset, timeattr=timeattr, traceattr=traceattr, activityattr=activityattr, base_folder = base_folder):
    train = LogFile("{}{}".format(base_folder, dataset), ",", 0, 30000, time_attr=timeattr, trace_attr=traceattr, activity_attr=activityattr, integer_input=False, convert=False)
    #train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    attrs = ['event', 'role', 'completeTime', 'case']
    train.remove_attributes([i for i in train.attributes() if i not in attrs])
    train.convert2int()
    model = cd.create_model(train, train)

    with open("model_30000b", "wb") as fout:
        pickle.dump(model, fout)

def experiment_standard(dataset, timeattr=timeattr, traceattr=traceattr, activityattr=activityattr, base_folder = base_folder):
    with open("model_30000b", "rb") as fin:
        model = pickle.load(fin)

    print("Get Scores")
    train = LogFile("{}{}".format(base_folder, dataset), ",", 0, 30000, time_attr=timeattr, trace_attr=traceattr, activity_attr=activityattr, integer_input=False, convert=False)
    #train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    attrs = ['event', 'role', 'completeTime', 'case']
    train.remove_attributes([i for i in train.attributes() if i not in attrs])
    train.convert2int()

    data = LogFile("{}{}".format(base_folder, dataset), ",", 0, None, time_attr=timeattr, trace_attr=traceattr, activity_attr=activityattr, convert=False, values=train.values, integer_input=False)
    #data.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    train.remove_attributes([i for i in train.attributes() if i not in attrs])
    data.convert2int()

    scores = cd.get_event_scores(data, model)
    cd.plot_single_scores(scores, base_folder)
    cd.plot_pvalues(scores, 1600, base_folder)


def experiment_attributes_standard():
    with open("model_30000b", "rb") as fin:
        model = pickle.load(fin)
    train = LogFile("../Data/BPIC18.csv", ",", 0, 30000, "startTime", "case", activity_attr=None, integer_input=False, convert=False)
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    train.convert2int()

    input = LogFile("../Data/BPIC18.csv", ",", 0, 10000000, "startTime", "case", convert=False, values=train.values)
    input.remove_attributes(["eventid", "identity_id", "event_identity_id", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    input.convert2int()

    data = input.filter_copy("self.data.year == 1")
    scores_year1 = cd.get_event_detailed_scores(data, model)
    cd.plot_attribute_graph(scores_year1, model.current_variables)

    data = input.filter_copy("self.data.year == 2")
    scores_year2 = cd.get_event_detailed_scores(data, model)
    cd.plot_attribute_graph(scores_year2, model.current_variables)

    data = input.filter_copy("self.data.year == 3")
    scores_year3 = cd.get_event_detailed_scores(data, model)
    cd.plot_attribute_graph(scores_year3, model.current_variables)

    p_vals_year1_2 = []
    p_vals_year2_3 = []
    p_vals_year1_3 = []
    for key in sorted(scores_year1.keys()):
        p_vals_year1_2.append(stats.ks_2samp(scores_year1[key], scores_year2[key]).pvalue)
        p_vals_year2_3.append(stats.ks_2samp(scores_year2[key], scores_year3[key]).pvalue)
        p_vals_year1_3.append(stats.ks_2samp(scores_year1[key], scores_year3[key]).pvalue)

    def tmp(x):
        if x == 0:
            return x
        else:
            return 1

    p_vals_year1_2 = [tmp(x) for x in p_vals_year1_2]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_2, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year2_3 = [tmp(x) for x in p_vals_year2_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year2_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year1_3 = [tmp(x) for x in p_vals_year1_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()

    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    for key in sorted(scores_year1.keys()):
        x.append(key)
        y_1.append(np.median(scores_year1[key]))
        y_2.append(np.median(scores_year2[key]))
        y_3.append(np.median(scores_year3[key]))
    plt.plot(x, y_1, "o")
    plt.plot(x, y_2, "o")
    plt.plot(x, y_3, "o")
    plt.xticks(rotation='vertical')
    plt.xlabel("Attributes")
    plt.ylabel("Median Score")
    plt.legend(["2015", "2016", "2017"])
    plt.show()

    p_vals_year1_2 = []
    p_vals_year2_3 = []
    p_vals_year1_3 = []
    for key in sorted(scores_year1.keys()):
        p_vals_year1_2.append(stats.ks_2samp(scores_year1[key], [np.median(scores_year1[key])]).pvalue)
        p_vals_year2_3.append(stats.ks_2samp(scores_year2[key], [np.median(scores_year2[key])]).pvalue)
        p_vals_year1_3.append(stats.ks_2samp(scores_year1[key], [np.median(scores_year3[key])]).pvalue)

    def tmp(x):
        if x == 0:
            return x
        else:
            return 1

    p_vals_year1_2 = [tmp(x) for x in p_vals_year1_2]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_2, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year2_3 = [tmp(x) for x in p_vals_year2_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year2_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()
    p_vals_year1_3 = [tmp(x) for x in p_vals_year1_3]
    plt.plot(sorted(scores_year1.keys()), p_vals_year1_3, "o")
    plt.xticks(rotation='vertical')
    plt.show()

def experiment_department():
    input = LogFile("../Data/BPIC18.csv", ",", 0, None, "startTime", "case", convert=False)
    input.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    input.convert2int()

    data = input.filter_copy("self.data.department == 1")
    model = cd.create_model(data, data)

    print("Starting writing model to file")
    with open("model_department", "wb") as fout:
        pickle.dump(model, fout)
    print("Done")

    with open("model_department", "rb") as fin:
        model = pickle.load(fin)

    for dept in [1, 2, 3, 4]:
        data = input.filter_copy("self.data.department == " + str(dept))
        scores = cd.get_event_detailed_scores(data, model)
        cd.plot_attribute_graph(scores, model.current_variables)

def experiment_clusters():
    with open("model_30000", "rb") as fin:
        model = pickle.load(fin)
    train = LogFile("../Data/BPIC18.csv", ",", 0, 30000, "startTime", "case", activity_attr=None, integer_input=False, convert=False)
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    train.convert2int()

    data = LogFile("../Data/BPIC18.csv", ",", 0, None, "startTime", "case", convert=False, values=train.values)
    data.remove_attributes(["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual",
                                    "penalty_amount", "risk_factor", "cross_compliance", "selected_random",
                                    "selected_risk", "selected_manually", "rejected"])
    data.convert2int()
    data.filter("self.data.year == 1")

    scores = cd.get_event_detailed_scores(data, model)

    # First calculate score per trace
    attributes = list(scores.keys())
    num_traces = len(scores[attributes[0]])
    upper = {}
    lower = {}
    for a in attributes:
        upper[a] = []
        lower[a] = []

    for trace_ix in range(num_traces):
        score = 1
        for a in scores:
            a_score = scores[a][trace_ix]
            if a_score == -5:
                score = 0
                break
            score *= a_score

        if -8 < score < -10:
            for a in scores:
                upper[a].append(scores[a][trace_ix])
        elif -10 < score < -12:
            for a in scores:
                lower[a].append(scores[a][trace_ix])
    print(attributes)
    print(upper)
    cd.plot_attribute_graph(upper, attributes)
    cd.plot_attribute_graph(lower, attributes)

def experiment_outliers():
    with open("model_30000", "rb") as fin:
        model = pickle.load(fin)
    train = LogFile("../Data/BPIC18.csv", ",", 0, 30000, "startTime", "case", activity_attr=None, integer_input=False, convert=False)
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    train.convert2int()

    attr_dicts = []

    data = LogFile("../Data/BPIC18.csv", ",", 0, None, "startTime", "case", convert=False, values=train.values)
    data.filter("self.data.year == 1")
    data.remove_attributes(["event_identity_id", "year", "penalty_", "amount_applied", "payment_actual",
                                    "penalty_amount", "risk_factor", "cross_compliance", "selected_random",
                                    "selected_risk", "selected_manually", "rejected"])

    scores = cd.get_event_scores(data, model)
    for s in scores:
        if sum(scores[s]) != 0:
            score = math.log10(sum(scores[s]) / len(scores[s]))
            if score < -12:
                for case in attr_dicts[0]:
                    if attr_dicts[0][case] == s:
                        print(s, case, score)


def analyze():
    train = LogFile("../Data/BPIC18.csv", ",", 0, None, "startTime", "case", activity_attr=None, integer_input=False, convert=False)
    print("Num of attributes:", len(train.data.columns))
    train.remove_attributes(["eventid", "identity_id", "event_identity_id", "year", "penalty_", "amount_applied", "payment_actual", "penalty_amount", "risk_factor", "cross_compliance", "selected_random", "selected_risk", "selected_manually", "rejected"])
    print("Num of attributes:", len(train.data.columns))
    print(train.data.columns)

    for attr in train.data.columns:
        print(attr, len(train.data[attr].value_counts()))

if __name__ == "__main__":
    dataset = 'Data/BPIC12.csv'

    learn_and_dump_model(dataset)
    experiment_standard(dataset)
    #experiment_attributes_standard()
    #experiment_department()
    #experiment_clusters()
    #experiment_outliers()
    # analyze()