import os
import sys
import Predictions.setting as setting
# import setting as setting
from Data.data import Data
from Utils.LogFile import LogFile
import Methods
import click as click
from pm4py.streaming.importer.xes import importer as stream_xes_importer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as stream_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
import PrefixTreeCDDmain.settings as settings
import time
import pandas as pd
from collections import deque
from itertools import islice
from collections import OrderedDict
from PrefixTreeCDDmain.CDD import Window
from skmultiflow.drift_detection import ADWIN, PageHinkley
import gzip


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_timings(file, timings):
    with open(file, "w") as fout:
        for t in timings:
            fout.write(str(t) + "\n")


@click.command()
@click.option('-l', '--decay_lambda', default=0.25, show_default=True, is_flag=False, flag_value=0.25,
              help='Decaying lambda for the older traces when performing a pruning step.')
@click.option('-n', '--noise', default=1, show_default=True, is_flag=False, flag_value=1,
              help='Noise filter applied to relations and activities which have a frequency below certain threshold.')
@click.option('-w', '--window_size', default=10, show_default=True, is_flag=False, flag_value=10,
              help='Maximum size of the window of detection (maxWindowSize)')
@click.option('-t', '--tree_size', default=1000, show_default=True, is_flag=False, flag_value=1000,
              help='Maximum size of the trees (pruningSteps)')
@click.option('-c', "--config", default=False, show_default=True, is_flag=False, flag_value=False,
              help='(True/False) Configuration for generating sub-logs from the drifts identified')
@click.option('-f', '--file', help='Path to the XES log file.')
def main(config, file='input_files/T_BPIC15_1.xes.gz', window_size=10, tree_size=1000, decay_lambda=0.25, noise=1):
    """This is the Prefix-Tree Concept Drift Detection algorithm."""
    DATASETS = ["BPIC15_1", "BPIC15_2"]
    # METHODS = ["SDL", "DBN", "DIMAURO", "TAX"]
    METHODS = ["TAX"]
    DRIFT = True
    # ADWIN = True
    RESET = [False, True]
    WINDOW = [0, 1, 5]
    batch = ["month"]

    configuration = config

    settings.init()
    endEventsDic = dict()
    window = Window(initWinSize=window_size)
    # filePath_gz = file
    # file_opened = gzip.open(file, 'rb')
    # filePath = file_opened

    # for file in os.listdir("dataset\BPI_Challenge_2020"):
    streaming_ev_object = stream_xes_importer.apply(os.path.abspath(file),
                                                    variant=stream_xes_importer.Variants.XES_TRACE_STREAM)

    # Process the log Trace-by-Trace
    for trace in streaming_ev_object:
        lastEvent = trace[-1]["concept:name"]
        timeStamp = trace[-1]["time:timestamp"]
        caseID = trace.attributes["concept:name"]
        endEventsDic[caseID] = [lastEvent, timeStamp]

    caseList = []  # Complete list of cases seen
    Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
    # print("You are here")
    tree = PrefixTree(pruningSteps=tree_size, noiseFilter=noise,
                      lambdaDecay=decay_lambda)  # Create the prefix tree with the first main node empty
    adwin = ADWIN()
    ph = PageHinkley()

    pruningCounter = 0  # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True,
                  variant.value.Parameters.TIMESTAMP_KEY: 'time:timestamp'}
    log = xes_importer.apply(os.path.abspath(file),
                             variant=variant)  # , parameters=parameters)

    static_event_stream = stream_converter.apply(log, variant=stream_converter.Variants.TO_EVENT_STREAM,
                                                 parameters=parameters)
    static_event_stream._list.sort(key=lambda x: x['time:timestamp'], reverse=False)

    eventCounter = 0  # Counter for number of events
    currentNode = tree.root  # Start from the root node

    start_time = time.time()
    # print("you are here 2")
    numEvents = 54818
    r = False
    histDf = pd.DataFrame(columns=['event', 'role', 'completeTime', 'case'])
    trainPer = 0.5
    driftsidd = [round(numEvents * trainPer)]
    driftID = False
    testedWins = [round(numEvents * trainPer)]
    lenTrainedActs = 0
    lenTrainedRoles = 0
    for ev in static_event_stream:
        # print(ev)
        # file_opened.close()
        # break
        event = {'event': ev["concept:name"], 'role': ev["org:resource"], 'completeTime': ev["time:timestamp"],
                 'case': ev["case:concept:name"]}
        histDf = histDf.append(event, ignore_index=True)
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, ev,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
        eventCounter += 1

        # initial training of the dataset
        if eventCounter == round(numEvents * trainPer):


            # d = Data('BPIC_2015', LogFile(histDf, time_attr="completeTime", trace_attr="case",
            #                               activity_attr='event', convert=False))
            #
            # # d = Data.get_data("BPIC15_1")
            # m = Methods.get_prediction_method("SDL")
            s = setting.STANDARD
            s.train_percentage = 100
            # #
            # d.prepare(s)
            # basic_model = m.train(d.train)
            # lenTrainedActs += len(d.train.values['event'])
            # lenTrainedRoles += len(d.train.values['role'])
            # print("Model Trained")
            # d.create_batch(b, timeformat)
            # results, timings = m.test_and_update_drift_adwin(basic_model, d, r)
            # store_results("results/%s_%s_adwin_update.csv" % (m.name, d.name), results)
            # store_timings("results/%s_%s_adwin_update_time.csv" % (m.name, d.name), timings)


        if window.cddFlag:  # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:  # Maximum size of window reached, start concept drift detection within the window
                window.conceptDriftDetection(adwin, ph)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)

                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree
                # else:
                #     driftID = True
            # if len(window.driftsIdentified) > len(driftsidd):
                # print("Drift identified at event {}".format(window.driftsIdentified, eventCounter))

        # if (len(driftsidd) - 1 < len(window.driftsIdentified) or eventCounter % 7000 == 0) and eventCounter > round(numEvents * trainPer):
        # if eventCounter % 7000 == 0 and eventCounter > round(
        #             numEvents * trainPer):
        # if (eventCounter % 5000 == 0 or eventCounter == numEvents) and eventCounter >= round(numEvents * trainPer):
            #Add in a new training to occur every 10k events automatically?
            # Do some stuff here to update the model with the newly identified drifts

            # print("Test Results Up to Drift")
            #
            # d = Data('BPIC_2015', LogFile(histDf[-20000:], time_attr="completeTime", trace_attr="case",
            #                               activity_attr='event', convert=False))
            # d.prepare(s)
            # print("Testing Previous Results")
            # d.train.contextdata = d.train.contextdata[testedWins[-1]:]
            # results = m.test(basic_model, d.train, lenTrainedActs, lenTrainedRoles)
            # store_results("workingFiles/SDL_5k_drift_reset.csv", results)
            # basic_model = m.train(d.train)
            # lenTrainedActs += len(d.train.values['event'])
            # lenTrainedRoles += len(d.train.values['role'])


            # print(d.train.contextdata)
            #speed this up, maybe use filters or exclusions or drops <-(probably fastest)
            # evList = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42]
            # roleList = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
            # #compl [4, 8, 12, 16, 20, 24, 28, 32, 36, 40] <- not applicable
            # cList = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
            #df = df[df.iloc[:, attrColumns].values < attrMaxVal]
            # attrList = [(evList, len(oldVals['event'])), (roleList, len(oldVals['role'])), (cList, len(oldVals['case']))]
            #to get max val, count length of list in d.train.values and determine max
            #trick from log context data creation
            # print("Shape of D Train context is {}".format(d.train.contextdata.shape))
            # print(d.train.contextdata)

            # for pair in attrList:
            #     for col in pair[0]:
            #         d.train.contextdata = d.train.contextdata[d.train.contextdata.iloc[:,col] < pair[1]]

            #print(d.train.contextdata)
            # print("Shape of D Train context is {}".format(d.train.contextdata.shape))
            # d.train.contextdata.to_csv(path_or_buf='workingFiles/dysfunctionalData.csv')
            # print("Testing Previous Results")
            # results = m.test(basic_model, d.train, lenTrainedActs, lenTrainedRoles)
            # store_results("workingFiles/SDL_OTF_4k_drift_reset.csv", results)
            # print("Updating model")

            # if len(driftsidd) != 0:
            #     # log_df = histDf[driftsidd[-1] + 1:]
            #     log_df = histDf[-4000:]
            # else:
                # log_df = histDf[-4000:]
            # log_df = histDf[-4000:]
            # oldVals = d.train.values
            # print(oldVals)
            # d = Data('BPIC_2015', LogFile(histDf[-10000:], time_attr="completeTime", trace_attr="case",
            #                               activity_attr='event', values=d.train.values, convert=False))
            # d.prepare(s)
            # #re-add new events to data (prep) before training again
            # basic_model = m.train(d.train)
            # lenTrainedActs = len(d.train.values['event'])
            # lenTrainedRoles = len(d.train.values['role'])
            # driftsidd.append(eventCounter)
            # driftID = False
    #This is still having problems (probably need to change driftsidd)
    # d = Data('BPIC_2015', LogFile(histDf[round(numEvents * trainPer):], time_attr="completeTime", trace_attr="case",
    #                               activity_attr='event', values=d.train.values, convert=False))
    # d.prepare(s)
    # results = m.test(basic_model, d.train, lenTrainedActs, lenTrainedRoles)
    # store_results("workingFiles/SDL_OTF_Base_BPIC15_1_0_drift_reset.csv", results)
    d = Data('BPIC_2015', LogFile(histDf, time_attr="completeTime", trace_attr="case",
                                  activity_attr='event', convert=False))
    d.prepare(s)
    saveFile = d.train.contextdata
    saveFile.to_csv(path_or_buf="BPIC15_CONTEXT.CSV")
    end_time = time.time()
    print("Total Processing Time {}".format(end_time - start_time))
    print("Total number of events is {}".format(eventCounter))
    print("Identified drifts {}".format(driftsidd))
    if configuration:
        logProvider(static_event_stream, window)  # Provide the sub-logs of the drifts


def as_dict(object):
    dictionaryObject = object.__dict__
    return dictionaryObject


def logProvider(log, window):
    """Function to provide the sub-logs of each drift"""
    for drift in window.driftsIdentified:
        W0start = drift.eventsSeen - drift.refWinSize
        W0end = drift.eventsSeen
        W1start = drift.eventsSeen
        W1end = drift.eventsSeen + drift.testWinSize

        W0SubLog = log[W0start:W0end]
        W1SubLog = log[W1start:W1end]

        if not os.path.exists("results"):
            os.makedirs("results")

        variables = W0SubLog[0].__dict__['_dict'].keys()

        df1 = pd.DataFrame([[i.get(j) for j in variables] for i in W0SubLog], columns=variables)
        df2 = pd.DataFrame([[i.get(j) for j in variables] for i in W1SubLog], columns=variables)

        indexOfDrift = str(window.driftsIdentified.index(drift))

        xes_exporter.apply(df1, os.path.abspath('results\\drift' + indexOfDrift + "W0.xes"))
        xes_exporter.apply(df2, os.path.abspath('results\\drift' + indexOfDrift + "W1.xes"))


if __name__ == "__main__":
    main(sys.argv[1:])
