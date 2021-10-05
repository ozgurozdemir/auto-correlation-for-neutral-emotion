import time
import numpy as np

from feature_utils import *
import tensorflow as tf
import librosa


def prepare_dataset(sig, lab, sr, batch_size=32):
    start = time.time()

    ## Mel Spectrogram
    X_spec = get_melspectrogram(sig, sr=sr)
    X_spec = tf.expand_dims(X_spec, -1)

    ## Autocorrelate
    X_auto = get_autocorrelate(sig, sr=sr)

    # dataset
    X_spec = tf.cast(X_spec, tf.float32)
    X_auto = tf.cast(X_auto, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((X_spec, X_auto, lab))
    ds = ds.shuffle(X_spec.shape[0]).batch(batch_size)

    print(f"Time spent for dataset preparation {time.time() - start} sec....")
    return ds


def train_step(model, spec, auto, labels, optimizer, loss_object, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(spec, auto)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


def train_model(epoch, train_ds, test_ds, model, loss_obj, optimizer,
                train_accuracy, train_loss, verbose=1):

    maxWA = 0
    maxUA = 0

    for epoch in range(epoch):
      start_time = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()

      ## training
      for step, (spec, auto, labels) in enumerate(train_ds):
        train_step(model, spec, auto, labels, loss_obj, optimizer, train_accuracy, train_loss)

      ## testing
      _, _, WA, UA, label_correct, _, label_acc = eval_model(model, test_ds, return_pred=True)

      # maximum results
      maxWA = WA if WA > maxWA else maxWA
      if (UA > maxUA):
          maxUA = UA
          print('** saving model (WA:{},UA:{})'.format(WA, UA))
          model.save_weights('./models/')

      ## print epoch info
      if verbose:
        template = "Epoch {}, Loss: {:.4f}, \033[92mAccuracy: {:.2f}\033[0m, Val Acc {:.2f}, \033[94mVal UA {:.2f}\033[0m - {:.4f} sec..\n"
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              WA * 100, UA * 100,
                              time.time() - start_time
                              ))
        print('label_correct:{}\nUA:{}'.format(label_correct, label_acc))
        print('maxWA:{}\nmaxUA:{}'.format(maxWA, maxUA))
        print('='*100)

    return model


def eval_model(model, test_ds, return_pred=False):
    label_correct = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    label_total = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    if return_pred:
      predictions = np.empty([1, 8])
      groundtruth = np.empty([1, 8])

    for _, (x, a, y) in enumerate(test_ds):
      out = model(x, a)

      if return_pred:
        predictions = np.concatenate([predictions, out.numpy()])
        groundtruth = np.concatenate([groundtruth, y.numpy()])

      out = tf.math.argmax(out, -1)
      y = tf.math.argmax(y, -1)

      # sum correct and total
      np.add.at(label_total, y.numpy(), 1)
      correct = out[out == y]
      np.add.at(label_correct, correct.numpy(), 1)

    label_acc = label_correct / label_total

    # average calculation
    WA = np.sum(label_correct) / np.sum(label_total)
    UA = np.mean(label_acc)

    if return_pred:
      return (groundtruth[1:], predictions[1:].argmax(axis=1),
              WA, UA,
              label_correct, label_total, label_acc)
    else:
      return WA, UA
