import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class IEMOCAP:
  """
  attr
    path:     str
    base:     sd (speaker-dependent) / si (speaker-independent)
    kind:     small(4 class) / small_merge(4+(1) class) / full
    modality: audio/text/mix
  """
  def __init__(self, path, base="si", kind="small_merge", modality="audio", seed=0):
    self.path=path
    self.base=base
    self.kind=kind
    self.seed=seed
    self.modality=modality
    self.merge_exc = (self.kind == "small_merge")

    if self.kind == "small" or self.kind == "small_merge":
      self.emo_classes = ["hap", "sad", "ang", "neu"]
      self.emo_one_hot = {"neu": [1, 0, 0, 0],
                          "sad": [0, 1, 0, 0],
                          "ang": [0, 0, 1, 0],
                          "hap": [0, 0, 0, 1]}
    else:
      raise("Full type was not implemented yet...")

    self.data = {}
    self.read_dataset()
    self.printInfo()


  def read_dataset(self):
    ses_id = ["Session1", "Session2", "Session3", "Session4", "Session5"]

    for ses in ses_id:
        self.data[ses] = self.read_session_data(ses)

    if self.base == "sd":
      self.data["X"] = np.concatenate([self.data["Session1"][0],
                                       self.data["Session2"][0],
                                       self.data["Session3"][0],
                                       self.data["Session4"][0],
                                       self.data["Session5"][0]])

      self.data["y"] = np.concatenate([self.data["Session1"][1],
                                       self.data["Session2"][1],
                                       self.data["Session3"][1],
                                       self.data["Session4"][1],
                                       self.data["Session5"][1]])
                                       
      del (self.data["Session1"], self.data["Session2"], self.data["Session3"],
           self.data["Session4"], self.data["Session5"])

  def read_session_data(self, ses_id):
    # Read description & find indices
    desc = pd.read_csv(self.path + ses_id + "_desc.csv")
    if self.merge_exc:
      desc["Emotion"].replace({"exc": "hap"}, inplace=True)

    idx = desc.index[desc["Emotion"].isin(self.emo_classes)].tolist() # indices of samples

    labels = desc.iloc[idx]["Emotion"].values
    labels = np.array([self.emo_one_hot[emo] for emo in labels])

    # Read signals
    waves = np.load(self.path + ses_id + "_waves.npz")["waves"]
    waves = waves[idx]

    # Read transcripts
    with open(self.path + ses_id + "_trans.csv", "r") as file:
      texts = file.read().split("\n")

    if self.modality == "audio":
      return waves, labels
    elif self.modality == "text":
      return texts, labels
    else:
      return waves, labels, texts


  def printInfo(self):
    print(f">> Base {self.base} experiments...")

    if self.base == "si":
      count = np.array([0, 0, 0, 0])
      for ses in self.data:
        _, y = self.data[ses]
        count += y.sum(axis=0)
        print(f":: {ses} - Utterance {y.sum()} - emo [neu, sad, ang, hap]: {y.sum(axis=0)}")
      print(f"Total {count.sum()} - emo [neu, sad, ang, hap]: {count}")

    elif self.base == "sd":
      print(f":: SEED: {self.seed}")
      _, trY, _, tsY = self.prepare_train_test_sets(seed=self.seed)
      print(f":: Total {self.data['y'].sum()} - emo [neu, sad, ang, hap]: {self.data['y'].sum(axis=0)}")
      print(f":: Train {trY.sum()} - emo [neu, sad, ang, hap]: {trY.sum(axis=0)}")
      print(f":: Test  {tsY.sum()} - emo [neu, sad, ang, hap]: {tsY.sum(axis=0)}")


  def prepare_train_test_sets(self, **args):
    if self.base == "si":
      train_set = args["train_set"]
      valid_set = args["valid_set"]
      test_set  = args["test_set"]

      if self.modality == "audio" or self.modality == "text":

        # train set
        trainX, trainY = self.data[train_set[0]]
        for i in range(1, len(train_set)):
          X, y = self.data[train_set[i]]
          trainX = np.append(trainX, X, axis=0)
          trainY = np.append(trainY, y, axis=0)

        # test set
        testX, testY = self.data[test_set[0]]
        for i in range(1, len(test_set)):
          X, y = self.data[test_set[i]]
          testX = np.append(testX, X, axis=0)
          testY = np.append(testY, y, axis=0)

        # valid set
        if valid_set is not None:
          validX, validY = self.data[valid_set[0]]
          return trainX, trainY, validX, validY, testX, testY
        else:
          return trainX, trainY, testX, testY

      else:
        raise("Mix modality was not implemented yet...")

    elif self.base == "sd":
      trainX, testX, trainY, testY = train_test_split(self.data["X"], self.data["y"],
                                                      test_size=0.2,
                                                      random_state=self.seed)
      if "valid_split" in args:
        validX, testX, validY, testY = train_test_split(testX, testY,
                                                        test_size=0.5,
                                                        random_state=self.seed)

        return trainX, trainY, validX, validY, testX, testY
      else:
        return trainX, trainY, testX, testY
