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

    # ------------CHANGE THIS TO ACCEPT INPUT FROM OTHER DATA SOURCE---------------------------------
    # d = Data.get_data("BPIC15_1")
    # m = Methods.get_prediction_method("SDL")
    # s = setting.STANDARD
    # s.train_percentage = 50
    #
    # d.prepare(s)
    # d.create_batch("normal", timeformat)
    # if m.name == "Di Mauro":
    #     m.def_params = {"early_stop": 4, "params": {"n_modules": 2}}
    #
    # import time
    # start_time = time.time()
    # dtrain = d.train.get_data
    # print(d.train.contextdata)
    # basic_model = m.train(d.train)
    #
    #
    # #----------COMMENCES TESTING OF MODEL TO BE BUILT INTO FOR LOOP---------------
    # print("Runtime %s:" % m, time.time() - start_time)
    # res = m.test(basic_model, d.test_orig)
    #
    # store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)
    #
    # d.create_batch(b, timeformat)
    # results, timings = m.test_and_update_drift_adwin(basic_model, d, r)
    # if r:
    #     store_results("results/%s_%s_adwin_reset.csv" % (m.name, d.name), results)
    #     store_timings("results/%s_%s_adwin_reset_time.csv" % (m.name, d.name), timings)
    # else:
    #     store_results("results/%s_%s_adwin_update.csv" % (m.name, d.name), results)
    #     store_timings("results/%s_%s_adwin_update_time.csv" % (m.name, d.name), timings)
    # --------------END TESTING CODE -------------------------------------------------------------
    numEvents = 54818
    r = False
    histDf = pd.DataFrame(columns=['event', 'role', 'completeTime', 'case'])
    driftsidd = []
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
        if eventCounter == round(numEvents / 2):
            # basic_model = m.train(d.train)
            # res = m.test(basic_model, d.test_orig)
            # store_results("results/%s_%s_normal.csv" % (m.name, d.name), res)
            print(histDf.head(5))

            d = Data('BPIC_2015', LogFile(histDf, time_attr="completeTime", trace_attr="case",
                                          activity_attr='event', convert=False))

            # d = Data.get_data("BPIC15_1")
            m = Methods.get_prediction_method("SDL")
            s = setting.STANDARD
            s.train_percentage = 100
            #
            d.prepare(s)
            basic_model = m.train(d.train)
            print("Model Trained")
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

        if len(driftsidd) < len(window.driftsIdentified):
            # Do some stuff here to update the model with the newly identified drifts

            driftsidd.append(eventCounter)

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
