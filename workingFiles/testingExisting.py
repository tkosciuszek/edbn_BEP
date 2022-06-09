import Predictions.setting
import Predictions.setting as setting
import Data
import Methods
from Utils.LogFile import LogFile
from Data.data import Data
import numpy as np

BASE_FOLDER = "/Users/teddy/Documents/TU_e/ThirdYear/Quartile_4/BEP/edbn_BEP/"

all_data = {"Helpdesk": BASE_FOLDER + "Data/Helpdesk.csv",
            "BPIC12": BASE_FOLDER + "Data/BPIC12.csv",
            "BPIC12W": BASE_FOLDER + "Data/BPIC12W.csv",
            "BPIC15_1": BASE_FOLDER + "Data/BPIC15_1_sorted_new.csv",
            "BPIC15_2": BASE_FOLDER + "Data/BPIC15_2_sorted_new.csv",
            "BPIC15_3": BASE_FOLDER + "Data/BPIC15_3_sorted_new.csv",
            "BPIC15_4": BASE_FOLDER + "Data/BPIC15_4_sorted_new.csv",
            "BPIC15_5": BASE_FOLDER + "Data/BPIC15_5_sorted_new.csv",
            "BPIC18": BASE_FOLDER + "Data/BPIC18.csv",
            "BPIC17": BASE_FOLDER + "Data/bpic17_test.csv",
            "BPIC19": BASE_FOLDER + "Data/BPIC19.csv",
            "BPIC11": BASE_FOLDER + "Data/BPIC11.csv",
            "SEPSIS": BASE_FOLDER + "Data/Sepsis.csv",
            "COSELOG_1": BASE_FOLDER + "Data/Coselog_1.csv",
            "COSELOG_2": BASE_FOLDER + "Data/Coselog_2.csv",
            "COSELOG_3": BASE_FOLDER + "Data/Coselog_3.csv",
            "COSELOG_4": BASE_FOLDER + "Data/Coselog_4.csv",
            "COSELOG_5": BASE_FOLDER + "Data/Coselog_5.csv",
            "Helpdesk2": BASE_FOLDER + "Data/helpdesk2.csv"}


def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_timings(file, timings):
    with open(file, "w") as fout:
        for t in timings:
            fout.write(str(t) + "\n")

def learn_model(log, attributes, epochs, early_stop):
    num_activities = len(log.values[log.activity]) + 1
    # Input + Embedding layer for every attribute
    input_layers = []
    embedding_layers = []
    for attr in attributes:
        if attr not in log.ignoreHistoryAttributes and attr != log.time and attr != log.trace:
            for k in range(log.k):
                i = Input(shape=(1,), name=attr.replace(" ", "_").replace("(", "").replace(")","").replace(":","_") + "_Prev%i" % k)
                input_layers.append(i)
                # e = Embedding(len(log.values[attr]) + 1, 32, embeddings_initializer="zeros")(i)
                e = Embedding(len(log.values[attr]) + 1, len(log.values[attr]) + 1, embeddings_initializer="zeros")(i)
                embedding_layers.append(e)
    concat = Concatenate()(embedding_layers)

    drop = Dropout(0.2)(concat)
    dense2 = Dense(num_activities)(drop)

    flat = Flatten()(dense2)

    output = Softmax(name="output")(flat)

    model = Model(inputs=input_layers, outputs=[output])
    opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=opt)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop)

    outfile = 'tmp/model_{epoch:03d}-{val_loss:.2f}.h5'
    model_checkpoint = ModelCheckpoint(outfile,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='auto')

    x, y, vals = transform_data(log, [a for a in attributes if a != log.time and a != log.trace])
    if len(y) < 10:
        split = 0
    else:
        split = 0.2
    model.fit(x=x, y=y,
              validation_split=split,
              verbose=2,
              callbacks=[early_stopping],
              batch_size=32,
              epochs=epochs)
    return model

#Method("SDL", sdl.train, sdl.test, sdl.update, {"epochs": 200, "early_stop": 10})
if __name__ == "__main__":
    # DATASETS = ["Helpdesk", "BPIC11", "BPIC12", "BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]
    DATASETS = ["BPIC15_1"]
    # METHODS = ["SDL", "DBN", "DIMAURO", "TAX"]
    METHODS = ["SDL"]
    DRIFT = True
    ADWIN = True
    RESET = [False, True]
    WINDOW = [0, 1, 5]
    batch = ["month"]
#Predefined Drift List from KNOWN Drift Locations Below
    DRIFT_LIST = {
        "Helpdesk": [9, 26],
        "BPIC11": [1, 9, 18],
        "BPIC12": [],
        "BPIC15_1": [2, 17, 24, 28],
        "BPIC15_2": [1, 7, 20, 27],
        "BPIC15_3": [1, 9, 15, 27],
        "BPIC15_4": [17, 20, 25],
        "BPIC15_5": [3, 20, 27]
    }

    for data_name in DATASETS:
        timeformat = "%Y-%m-%d %H:%M:%S"
        if "BPIC15" not in data_name:
            timeformat = "%Y/%m/%d %H:%M:%S.%f"


        for m in METHODS:
            sep = ","
            time = "completeTime"
            case = "case"
            activity = "event"
            resource = "role"
            # d = Data.get_data(data_name)
            # m = Methods.get_prediction_method(m)
            # s = setting.STANDARD
            # s.train_percentage = 50
            logf = LogFile(all_data[data_name], sep, 0, None, time, case, activity_attr=activity, convert=False)
            logf2 = logf.get_data()
            print(logf2.columns)
            print(logf2)
            break



