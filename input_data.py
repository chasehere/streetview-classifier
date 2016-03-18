import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.image as mpimg

def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

class DataSet(object):
  def __init__(self, images, labels, ids):
    assert images.shape[0]==labels.shape[0]==ids.shape[0], (
        "images.shape: %s, labels.shape: %s, ids.shape: %s"
        % (images.shape,labels.shape.ids.shape))

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._ids = ids
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_compelted(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # finished epoch
      self._epochs_completed += 1

      # shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      self._ids = self._ids[perm]

      #start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples

    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(train_dir):
  np.random.seed(1234) # for repeatable experiments
  
  class DataSets(object):
    pass

  data_sets = DataSets()

  # load the data
  images, labels, ids = load_images(train_dir)

  test_images = images[np.where(labels == 2)[0]][:] 
  test_labels = labels[np.where(labels == 2)[0]][:]
  test_ids = ids[np.where(labels == 2)[0]][:]

  # number of valid samples per class
  num_positive_samples = len(labels[labels == 1])
  num_negative_samples = len(labels[labels == 0])
  num_pos_valid_samples = int(.35 * num_positive_samples)
  num_neg_valid_samples = num_pos_valid_samples
  
  idx_pos_valid = np.random.choice( num_positive_samples, size = num_pos_valid_samples, replace=False) 
  idx_neg_valid = np.random.choice( num_negative_samples, size = num_neg_valid_samples, replace=False)
  
  validation_images = np.concatenate(( 
    images[ np.where(labels==1)[0][idx_pos_valid]][:],
    images[ np.where(labels==0)[0][idx_neg_valid]][:]))  
  validation_labels = np.concatenate((
    labels[ np.where(labels==1)[0][idx_pos_valid]][:],
    labels[ np.where(labels==0)[0][idx_neg_valid]][:]))  
  validation_ids = np.concatenate(( 
    ids[ np.where(labels==1)[0][idx_pos_valid]][:],
    ids[ np.where(labels==0)[0][idx_neg_valid]][:]))  

  idx_pos_train = [n for n in range(num_positive_samples) if n not in idx_pos_valid]
  idx_neg_train = [n for n in range(num_negative_samples) if n not in idx_neg_valid]
 
  train_images = np.concatenate((
    images[ np.where(labels==1)[0][idx_pos_train]][:],
    images[ np.where(labels==0)[0][idx_neg_train]][:]))
  train_labels = np.concatenate((
    labels[ np.where(labels==1)[0][idx_pos_train]][:],
    labels[ np.where(labels==0)[0][idx_neg_train]][:]))
  train_ids = np.concatenate((
    ids[ np.where(labels==1)[0][idx_pos_train]][:],
    ids[ np.where(labels==0)[0][idx_neg_train]][:]))

  data_sets.train = DataSet(train_images, train_labels, train_ids)
  data_sets.validation = DataSet(validation_images, validation_labels, validation_ids)
  data_sets.test = DataSet(test_images, test_labels, test_ids)
  return data_sets

def load_images(train_dir):
  print "Loading images..."
  start = time.time()
  total_images = len([filename for filename in os.listdir(train_dir)]) 
  
  images = np.zeros([total_images, 128*128])
  labels = np.zeros([total_images, 1])
  ids = np.zeros([total_images, 1])

  counter = 0
  start2 = time.time()

  missing_img = mpimg.imread('images/450316303013000024_0.jpg')
  
  for filename in os.listdir(train_dir):
    # read image
    img = mpimg.imread('images/' + filename)
    if np.array_equal(img,missing_img):
      continue
    
    counter += 1    

    # save meta data
    ids[counter-1] = filename.split('_')[0]
    labels[counter-1] = filename.split('_')[1][0]
 
    
    # convert to gray scale (TODO: try keeping rgb)
    img = rgb2gray(img)

    # flatten and scale (TODO: try differing scaling besides 0-1)
    img = img.flatten()
    img = img / 255.0
    images[counter-1,:] = img

    # print updates
    if( counter % 500 == 0):
      print "Processed %s images, Time/batch = %.3f, Total Time = %.3f" % (counter, time.time() - start2, time.time() - start)
      start2 = time.time()

  print "Completed %s images, Total Time = %.3f" % (counter, time.time() - start)
  return images, labels, ids
