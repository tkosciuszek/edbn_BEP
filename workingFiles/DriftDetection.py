import time
import json
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


def run(file, dataName, wSize = 10, tSize = 1000, dLamb = 0.25, noise = 1):
    file = file

    data = pd.read_csv(file, low_memory=False)

    numEvents = data.shape[0]
    print("Num events is {}".format(numEvents))







    window_size = wSize
    tree_size = tSize
    decay_lambda = dLamb
    noise = noise
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

    drifts = {}
    sTime = time.time()
    for _, event in data.iterrows():

        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, event,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
        eventCounter += 1

        if window.cddFlag:
            if len(window.prefixTreeList) == window.WinSize:

                temp_drifts = window.conceptDriftDetection(adwin, ph, eventCounter)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)
                for i in temp_drifts.keys():
                    if i not in drifts.keys():
                        drifts[i] = temp_drifts[i]

    drifts['Time'] = time.time() - sTime
    with open("results/driftJSON/{}.json".format(dataName), "w") as fp:
        json.dump(drifts, fp)




if __name__ == "__main__":
    winS = [8, 10, 12]
    preT = [500, 800, 1000]
    lamb = [0, 0.15, 0.25]
    files = ['Data/BPIC11.csv', 'Data/BPIC12.csv', 'Data/BPIC15_2_sorted_new.csv', 'Data/BPIC15_3_sorted_new.csv',
             'Data/BPIC15_4_sorted_new.csv']
    #File format is DataName_MaxWinSize_PrefixTreeSize_DecayLambda.json
    names = ['BPIC11', 'BPIC12', 'BPIC15_2', 'BPIC15_3', 'BPIC15_4']
    for l in lamb:
        for w in winS:
            for t in preT:
                for f in range(len(files)):
                    start = time.time()
                    print("Beginning {}_{}_{}_{}".format(names[f], w, t, l))
                    run(file = files[f], dataName="{}_{}_{}_{}".format(names[f], w, t, l),
                        wSize = w, tSize = t, dLamb=l)
                    print("Took {} seconds to complete {}_{}_{}_{}".format(time.time() - start, names[f], w, t, l))


