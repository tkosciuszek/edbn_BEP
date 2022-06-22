import time
import pandas as pd
from collections import deque
from itertools import islice
import os
import sys
import Predictions.setting as setting
from collections import OrderedDict
from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
import PrefixTreeCDDmain.settings as settings
from PrefixTreeCDDmain.CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley

data = pd.read_csv("Data/BPIC15_ALL_context.csv")

window_size=10
tree_size=1000
decay_lambda=0.25
noise=1
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

start_time = time.time()
for _, event in data.iterrows():
    #need to implement ev into form accepted from event in line above
    caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, event,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
    eventCounter += 1

    if window.cddFlag:  # If a complete new tree has been created
        if len(window.prefixTreeList) == window.WinSize:  # Maximum size of window reached, start concept drift detection within the window
            window.conceptDriftDetection(adwin, ph)
            window.WinSize = min(window.WinSize + 1, window.maxWindowSize)

            if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree

end_time = time.time()
print("Total Processing Time {}".format(end_time - start_time))
print("Total number of events is {}".format(eventCounter))