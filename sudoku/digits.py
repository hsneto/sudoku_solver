# TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
from os import mkdir, listdir, remove
from os.path import exists, join
from cv2 import resize
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

############################### LOAD DATASET #########################

def load_digits_data(filename='./dataset/digits-dataset'):
  """
  Load the dataset in training and testing sets
  """
  with open(filename, 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)
  return x_train, y_train, x_test, y_test

############################### LAYERS ###############################

def conv_layer(input_x, shape, name='conv_layer'):
  """
  Create convolutional layer
  """
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
    b = tf.Variable(tf.constant(0.1, shape=[shape[3]]), name='B')
    conv = tf.nn.conv2d(input_x, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
  
def load_conv_layer(input_x, w, b, name='conv_layer'):
  """
  Load convolutional layer based on already trained weights and bias
  """
  with tf.name_scope(name):
    conv = tf.nn.conv2d(input_x, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def fc_layer(input_x, size_in, size_out, name='fc_layer'):
  """
  Create fully-connected layer
  """
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input_x, w) + b
    
    tf.summary.histogram('weights', w)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    
    return act, w

def load_fc_layer(input_x, w, b, name='fc_layer'):
  """
  Load fully-connected layer based on already trained weights and bias
  """
  with tf.name_scope(name):
    act = tf.matmul(input_x, w) + b
    return act, w 

############################### TRAINING ##############################

def train_model(learning_rate, steps=2000, dropout=0.5, batch_size=50 , logdir=None, save_model=False, dataset_dir='./dataset/digits-dataset'):
  """
  Train CNN to recognise digits [0-9]

  Args:
    learning_rate --> Learning rate schedule for weight updates;
    steps         --> Number of epochs;
    dropout       --> Dropout value. 0. to discard all neurons and 1. to keep them;
    batch_size    --> Batch size of mini-batch SGD;
    logdir        --> Directory to save tensorboard events;
    save_model    --> If true, the model will be saves at /model/digits_model.ckpt
    dataset_dir   --> Dataset directory used to train the CNN
  """

  # Load dataset
  x_train, y_train, x_test, y_test = load_digits_data(dataset_dir)

  tf.reset_default_graph()
  sess = tf.Session()
  
  # Placeholders and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
  keep_prob = tf.placeholder(tf.float32, name='keep_prob')
  
  x_image = tf.reshape(x, [-1,28,28,1])
  tf.summary.image('input', x_image, 3)
  
  # Convolutional and pooling layers  
  conv1 = conv_layer(x_image, shape=[5,5,1,32], name='conv1')
  conv2 = conv_layer(conv1, shape=[5,5,32,64], name='conv2')
  
  # Fully connected layers and dropout
  flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
  
  fc1 = fc_layer(flattened, 7*7*64, 1000, name='fc1')
  relu = tf.nn.relu(fc1[0])
  
  fc1_drop = tf.nn.dropout(relu, keep_prob=keep_prob, name='dropout')
  
  fc2 = fc_layer(fc1_drop, 1000, 10, 'fc2')               
  logits = fc2[0]
  
  tf.summary.histogram('fc1/relu', relu)
  
  # Loss function
  with tf.name_scope('xent'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y), 
        name='xent')
    
    tf.summary.scalar('xent', cross_entropy)
 
  # Optimizer
  with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
  # Accuracy
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
  
  summ = tf.summary.merge_all()

  # Add ops to save and restore all the variables.
  saver = tf.train.Saver()
  
  sess.run(tf.global_variables_initializer())

  if logdir is not None:
    if not exists(logdir):
      mkdir(logdir)
    else:
      # clear dir
      filelist = [ f for f in listdir(logdir)]
      for f in filelist:
        remove(join(logdir, f))
  else:
    logdir = '/tmp/'

  writer = tf.summary.FileWriter(logdir)
  writer.add_graph(sess.graph)
  
  for i in range(steps):
    batch_idx = np.random.choice(9000, batch_size, replace=True)
    batch_x = x_train[batch_idx]
    batch_y = y_train[batch_idx]

    if i % 5 == 0:
      [_, s] = sess.run([accuracy, summ], feed_dict={
          x: batch_x, y: batch_y, keep_prob:1.0})
      writer.add_summary(s, i)
      
    if i % 100 == 0:
      print('train accuracy on step {}: {}'.format(
          i, sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0})))
            
    sess.run(train_op, feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})
    
  print('test accuracy: ', sess.run(accuracy, feed_dict={
      x: x_test, y: y_test, keep_prob:1.0}))

  # Save the variables to disk.
  if save_model:
    save_dir = './model/'

    if not exists(save_dir):
      mkdir(save_dir)
    else:
      # clear dir
      filelist = [ f for f in listdir(save_dir)]
      for f in filelist:
        remove(join(save_dir, f))
        
    save_path = saver.save(sess, "./model/digits_model.ckpt")
    print("Model saved in path: %s" % save_path)
      
############################### PREDICT ###############################

def process_data(data):
  """
  Pre-process data to feed the predictor.
  """

  # resize input image
  f1 = lambda x : resize(x, (28, 28))
  # reshape input image (list)
  f2 = lambda x : x.reshape(-1)
  # reshape input image
  f3 = lambda x : x.reshape(1, -1)

  if isinstance(data, np.ndarray):
    digits = f3(data)
  elif isinstance(data, list):
    digits = [f2(x) for x in data]
  else:
    print('Invalid input data!')
    return

  return np.asarray(digits) / 255

def predict(data, preprocess_data=True, model_dir='./model/'):
  """
  Predict data classes based on a trained model

  Args:
    data            --> Image(s) which will be recognised.
    preprocess_data --> Normalize data and pack it as a np.ndarray which each row is one image
    model_dir       --> Directory that contains the trained model (ckpt files)
  """

  tf.reset_default_graph()
  sess=tf.Session()    

  # First let's load meta graph and restore weights
  meta_file = [f for f in listdir('./model/') if f.endswith('.meta')][0]
  saver = tf.train.import_meta_graph(join(model_dir, meta_file))
  saver.restore(sess,tf.train.latest_checkpoint(model_dir))
  print("Model restored.")

  # Placeholders
  x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
  x_image = tf.reshape(x, [-1,28,28,1])

  # Load weights
  wc1, bc1 = sess.run(['conv1/W:0', 'conv1/B:0'])
  wc2, bc2 = sess.run(['conv2/W:0', 'conv2/B:0'])
  wf1, bf1 = sess.run(['fc1/W:0', 'fc1/B:0'])
  wf2, bf2 = sess.run(['fc2/W:0', 'fc2/B:0'])

  # Load Convolutional and pooling layers  
  conv1 = load_conv_layer(x_image, wc1, bc1, name='conv1')
  conv2 = load_conv_layer(conv1, wc2, bc2, name='conv2')

  # Load Fully connected layers
  flattened = tf.reshape(conv2, [-1, 7 * 7 * 64])
  fc1 = load_fc_layer(flattened, wf1, bf1, name='fc1')
  relu = tf.nn.relu(fc1[0])
  fc2 = load_fc_layer(relu, wf2, bf2, name='fc2')
  logits = fc2[0]

  # Prediction
  prediction = tf.argmax(logits, 1)

  # Process data
  if preprocess_data:
    digits = process_data(data)
  else:
    digits = data

  # Run 
  sess.run(tf.global_variables_initializer())
  out = sess.run(prediction, feed_dict={x:digits})

  return out