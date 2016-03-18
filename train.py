import tensorflow as tf
import input_data


def main(epochs=10000, batch_size=150, learn_rate=0.01):
  data = input_data.read_data_sets('images')

  # Placeholders  
  x = tf.placeholder("float",[None, 128*128])
  y = tf.placeholder("float", [None, 1])

  # Variables
  W = tf.Variable( tf.zeros([128*128,1]) )
  b = tf.Variable( tf.zeros([1]) )

  y_pred = tf.nn.sigmoid(tf.matmul(x,W) + b)

  # helper functions
  correct_prediction = tf.equal(tf.round(y_pred), y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
  # Optimization
  cost = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(y_pred, y))
  #train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
  train_step = tf.train.MomentumOptimizer(.003, .5).minimize(cost)

  init = tf.initialize_all_variables()

  sess = tf.Session()
  sess.run(init)

  # Initialize
  for epoch in range(epochs):
    batch_xs, batch_ys = data.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
    
    if epoch % 100 == 0:
      train_cost = sess.run(cost, feed_dict={x: data.train.images, y: data.train.labels}) / data.train.num_examples
      valid_cost = sess.run(cost, feed_dict={x: data.validation.images, y: data.validation.labels}) / data.validation.num_examples
      train_acc = sess.run(accuracy, feed_dict={x: data.train.images, y: data.train.labels})
      valid_acc = sess.run(accuracy, feed_dict={x: data.validation.images, y: data.validation.labels})
      print "epoch %s, train/valid cost: %.3f/%.3f, train/valid accuracy %.3f/%.3f" % (epoch, train_cost, valid_cost, train_acc, valid_acc)

  sess.close()


if __name__ == "__main__":
  main()

