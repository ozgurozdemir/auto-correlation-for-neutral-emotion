import time
import numpy as np
import pandas as pd

from models import *
from datatet_utils import *
from feature_utils import *
from training_utils import *

from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def test_experiment(model, test_dataset, scores=None):
  (groundtruth, predictions,
   WA, UA,
   label_correct, label_total, label_acc) = eval_model(model, test_dataset, return_pred=True)

  groundtruth = np.argmax(groundtruth, -1)
  # predictions = np.argmax(predictions, 0)

  info = (f"WA: {WA}\nUA: {UA}\nF1: {f1_score(groundtruth, predictions, average='weighted')}\n" +
          f"\n               [neu sad ang hap]" +
          f"\nLabel total: {label_total}\nLabel correct: {label_correct}\nLabel acc: {label_acc}\n\n" + "-"*45 +
          f"\nConfusion mat:\n{str(confusion_matrix(groundtruth, predictions))}\n\n" + "-"*45 +
          f"\nConfusion perc mat:\n{str(confusion_matrix(groundtruth, predictions, normalize='true'))}\n")

  return info, scores


def make_experiments_cv(dataset_path, experiment_path, seeds, include_auto=True):
  if not os.path.exists(experiment_path):
    raise "Experiment path does not exist..."
  if not os.path.exists(dataset_path):
    raise "Dataset path does not exist..."

  sr = 48000 if "ravdess" in dataset_path else 16000
  num_class = 8 if "ravdess" in dataset_path else 4

  for seed in seeds:
    start_time = time.time()

    print(f"Training for -{seed}- starts...")

    ## dataset preparation
    if "ravdess" in dataset_path:
        dataset = np.load(dataset_path)
        X, y    = dataset['X'], dataset['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=seed)
    else:
        dataset = IEMOCAP(dataset_path, base="sd", seed=seed)
        X_train, y_train, X_test, y_test = dataset.prepare_train_test_sets()

    train_ds = prepare_dataset(X_train, y_train, sr)
    test_ds  = prepare_dataset(X_test,  y_test,  sr)

    ## model initialize
    model = Combine_CNN_RNN(num_class, 16, 0.5, advance=False, include_auto=include_auto)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    ## train
    train_model(100, train_ds, test_ds, model, optimizer, loss_object, train_loss, train_accuracy, verbose=0)

    ## test
    model = Combine_CNN_RNN(num_class=8, units=16, dropout=0.5, advance=False, include_auto=include_auto)
    model(sample_inp_spec, sample_inp_auto)
    model.load_weights("chkp.h5")
    info, _ = test_experiment(model, test_ds, None)

    path = experiment_path + f"auto_{seed}.txt" if include_auto else experiment_path + f"{seed}.txt"
    with open(path, "w") as file:
      file.write(info)
    print(f"Experiment -{seed}- is ended in {time.time()-start_time} sec...\n")


if __name__ == "__main__":
    SEEDS = [42, 123456, 424242, 777777, 999999]
    DATA_PATH = "dataset/iemocap"
    EXP_PATH  = "experiments/"
    USE_AUTO  = True

    make_experiments_cv(DATA_PATH, EXP_PATH, SEEDS, USE_AUTO)
