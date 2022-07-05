import time
import pandas as pd
from Data.data import Data
from Utils.LogFile import LogFile
import Predictions.setting as setting
import Methods
from collections import deque
from itertools import islice
from collections import OrderedDict
from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
import PrefixTreeCDDmain.settings as settings
from PrefixTreeCDDmain.CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_resultsa(file, results):
    with open(file, "a") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_timings(file, timings):
    with open(file, "w") as fout:
        for t in timings:
            fout.write(str(t) + "\n")


def store_timing(file, timing):
    with open(file, "w") as fout:
        fout.write(str(timing) + "\n")


def store_timinga(file, timing):
    with open(file, "a") as fout:
        fout.write(str(timing) + "\n")


def run(file, dataName, updateInt, updateWindow, trainPerc=0.5):
    file = file
    # file = "Data/BPIC15_1_sorted_new.csv"
    # file = "Data/BPIC15_ALL.csv"
    # data = pd.read_csv("Data/BPIC15_1_sorted_new.csv", low_memory=False)
    data = pd.read_csv(file, low_memory=False)
    timeformat = "%Y-%m-%d %H:%M:%S"
    numEvents = data.shape[0]
    print("Num events is {}".format(numEvents))

    d = Data(dataName,
             LogFile(filename=file, delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                     activity_attr='event', convert=False))
    d.logfile.keep_attributes(['event', 'role', 'completeTime'])
    m = Methods.get_prediction_method("SDL")
    s = setting.STANDARD
    trainPerc = trainPerc
    s.train_percentage = trainPerc * 100
    # # #
    d.prepare(s)
    # d.create_batch("normal", timeformat)
    is_written = 0

    start_time = time.time()
    # print(d.train.contextdata)
    print("Test Context Data")
    print(d.test_orig.contextdata)
    basic_model = m.train(d.train)

    print("Runtime %s:" % m, time.time() - start_time)
    res = m.test(basic_model, d.test_orig)

    store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)
    b = 'month'
    # d.create_batch(b, timeformat)

    print("Baseline Complete")
    updateInt = updateInt
    updateWindow = updateWindow
    window_size = 10
    tree_size = 1000
    decay_lambda = 0.25
    noise = 1
    settings.init()
    endEventsDic = dict()
    window = Window(initWinSize=window_size)

    lastEvents = data.groupby(['case']).last()
    for _, row in lastEvents.iterrows():
        endEventsDic[_] = [str(row['event']), row['completeTime']]

    caseList = []  # Complete list of cases seen
    Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
    # print("You are here")
    tree = PrefixTree(pruningSteps=tree_size, noiseFilter=noise,
                      lambdaDecay=decay_lambda)  # Create the prefix tree with the first main node empty
    adwin = ADWIN()
    ph = PageHinkley()

    pruningCounter = 0  # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model

    eventCounter = 0  # Counter for number of events
    currentNode = tree.root  # Start from the root node
    testInd = round(numEvents * trainPerc)
    drifts = {}
    start_time = time.time()
    for _, event in data.iterrows():
        # print(event)
        # break
        # need to implement ev into form accepted from event in line above
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, event,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
        eventCounter += 1

        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:
                # Maximum size of window reached, start concept drift detection within the window
                temp_drifts = window.conceptDriftDetection(adwin, ph, eventCounter)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)
                for i in temp_drifts.keys():
                    if i not in drifts.keys():
                        drifts[i] = temp_drifts[i]
                        # Retrain model or smth here with condition index > train count
                        # Need to update the testing batch so it tests all the way back to testInd
                        if _ > round(numEvents * trainPerc):
                            print("Performing Drift Update")
                            result, timing, basic_model = m.test_and_update_indices(basic_model, d,
                                                                                    (_ - round(numEvents * trainPerc),
                                                                                     max(_ - round(
                                                                                         numEvents * trainPerc) - updateWindow,
                                                                                         0)),
                                                                                    testInd - round(
                                                                                        numEvents * trainPerc),
                                                                                    _ - round(numEvents * trainPerc),
                                                                                    reset=True)
                            if is_written:
                                store_resultsa(
                                    "results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow),
                                    result)
                                store_timinga(
                                    "results/%s_%s_OTF_drift_%s_%s_time.csv" % (
                                    m.name, d.name, updateInt, updateWindow),
                                    timing)
                            else:
                                store_results(
                                    "results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow),
                                    result)
                                store_timing(
                                    "results/%s_%s_OTF_drift_%s_%s_time.csv" % (
                                    m.name, d.name, updateInt, updateWindow),
                                    timing)
                            is_written = 1
                            testInd = _
                            print("Model Drift updated at event {}".format(str(_)))

                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

        # Troubleshooting Print statements for every x events
        # if _ % 10000 == 0:
        #     print("10000 events have passed")
        #     print("Hyphen")
        #     print(_)
        #     print("TestInd")
        #     print(_)
        #     print("Since last test")
        #     print(_ - testInd)
        #     print("Train size")
        #     print(round(numEvents * trainPerc))

        if (_ - testInd > updateInt) and (_ > round(numEvents * trainPerc)):
            print("Performing Maintenance Update")
            print("Size of hyphen is {}".format(_))
            print("Highest index calculation on test is {}".format(_ - round(numEvents * trainPerc)))
            result, timing, basic_model = m.test_and_update_indices(basic_model, d, (_ - round(numEvents * trainPerc),
                                                                                     max(_ - round(
                                                                                         numEvents * trainPerc) - updateWindow,
                                                                                         0)),
                                                                    testInd - round(numEvents * trainPerc),
                                                                    _ - round(numEvents * trainPerc),
                                                                    reset=False)
            testInd = _
            print("Model updated at event {}".format(str(_)))
            if is_written:
                store_resultsa("results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow), result)
                store_timinga("results/%s_%s_OTF_drift_%s_%s_time.csv" % (m.name, d.name, updateInt, updateWindow),
                              timing)
            else:
                store_results("results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow), result)
                store_timing("results/%s_%s_OTF_drift_%s_%s_time.csv" % (m.name, d.name, updateInt, updateWindow),
                             timing)
            is_written = 1

    print("Performing Final Test")
    # print("Size of hyphen is {}".format(_))
    # print("Highest index calculation on test is {}".format(_ - round(numEvents * trainPerc)))
    result, timing, basic_model = m.test_and_update_indices(basic_model, d, (_ - round(numEvents * trainPerc),
                                                                             max(_ - round(
                                                                                 numEvents * trainPerc) - updateWindow,
                                                                                 0)),
                                                            testInd - round(numEvents * trainPerc),
                                                            _ - round(numEvents * trainPerc),
                                                            reset=False)
    testInd = _

    print("Model updated at event {}".format(str(_)))
    if is_written:
        store_resultsa("results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow), result)
        store_timinga("results/%s_%s_OTF_drift_%s_%s_time.csv" % (m.name, d.name, updateInt, updateWindow), timing)
    else:
        store_results("results/%s_%s_OTF_drift_%s_%s.csv" % (m.name, d.name, updateInt, updateWindow), result)
        store_timing("results/%s_%s_OTF_drift_%s_%s_time.csv" % (m.name, d.name, updateInt, updateWindow), timing)

    end_time = time.time()


if __name__ == "__main__":
    intervals = [200]#, 500, 1000, 5000]
    windows = [200]#, 500, 1000, 5000]
    files = ['Data/BPIC12.csv']
             #'Data/BPIC15_2_sorted_new.csv', 'Data/BPIC15_3_sorted_new.csv', 'Data/BPIC15_5_sorted_new.csv']
    names = ['BPIC12']
    for i in intervals:
        for w in windows:
            for f in range(len(files)):
                if i < 1000 and w < 1000:
                    run(files[f], names[f], i, w)
                elif i >= 1000:
                    run(files[f], names[f], i, w)
