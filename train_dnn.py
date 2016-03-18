import tensorflow as tf
import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def main(epochs=1000, batch_size=50, learn_rate=1e-4, dropout=0.5):
  data = input_data.read_data_sets('images')

  # Placeholders
  x = tf.placeholder(tf.float32,[None, 128*128])
  y = tf.placeholder(tf.float32, [None, 1])

  # Variables
  W = tf.Variable( tf.zeros([128*128,1]) )
  b = tf.Variable( tf.zeros([1]) )
  
  # Target variable
  #y_pred = tf.nn.sigmoid(tf.matmul(x,W) + b)

  print 'checkpoint 1'
  # Deep model graph - first convolutional layer
  W_conv1 = weight_variable([5,5,1,32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,128,128,1])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  # Deep model graph - second convolutional layer
  print 'checkpoint 2'
  W_conv2 = weight_variable([5,5,32,64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # Densely connected layer
  print 'checkpoint 3'
  W_fc1 = weight_variable([32 * 32 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout	
  print 'checkpoint 4'
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


  # Readout layer
  print 'checkpoint 5'
  W_fc2 = weight_variable([1024,1])
  b_fc2 = bias_variable([1])
  y_pred = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  # helper functions
  correct_prediction = tf.equal(tf.round(y_pred), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  # Optimization
  cost = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y))
  #train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
  #train_step = tf.train.MomentumOptimizer(.003, .5).minimize(cost)
  train_step = tf.train.AdamOptimizer(learn_rate).minimize(cost)

  # Initialization
  print 'checkpoint 6'
  init = tf.initialize_all_variables()
  sess = tf.Session()
  sess.run(init)
  
  # Gradient descent over model graph
  print 'Beginning stoch gradient descent...'
  for epoch in range(epochs):
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
    
    if epoch % 100 == 0:
      train_cost = sess.run(cost, feed_dict={x: data.train.images, y: data.train.labels, keep_prob: 1.0}) / data.train.num_examples
      valid_cost = sess.run(cost, feed_dict={x: data.validation.images, y: data.validation.labels, keep_prob: 1.0}) / data.validation.num_examples
      train_acc = sess.run(accuracy, feed_dict={x: data.train.images, y: data.train.labels, keep_prob: 1.0})
      valid_acc = sess.run(accuracy, feed_dict={x: data.validation.images, y: data.validation.labels, keep_prob: 1.0})
      print "epoch %s, train/valid cost: %.3f/%.3f, train/valid accuracy %.3f/%.3f" % (epoch, train_cost, valid_cost, train_acc, valid_acc)

  sess.close()

if __name__ == "__main__":
  main()

