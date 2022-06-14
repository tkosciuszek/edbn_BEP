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
def main(config, file, window_size=10, tree_size=1000, decay_lambda=0.25, noise=1):
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
    print("I actually read this line")
    streaming_ev_object = stream_xes_importer.apply(os.path.abspath(file),
                                                        variant=stream_xes_importer.Variants.XES_TRACE_STREAM)

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
        if eventCounter == round(numEvents / 4):
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
            print(d.train.df)

if __name__ == "__main__":
    main(sys.argv[1:])