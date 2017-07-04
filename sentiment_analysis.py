import numpy as np
import tensorflow as tf
from string import punctuation
from collections import Counter


# Hyperparameters
lstm_size = 256
lstm_layers = 1
batch_size = 256
learning_rate = 0.001
epochs = 10
keep_probability = 0.5
version = '1'
summary_path = './logs/' + version

# Load data

with open('./reviews.txt', 'r') as f:
    reviews = f.read()

with open('./labels.txt', 'r') as f:
    labels = f.read()


# Data preprocessing

all_text = ''.join([c for c in reviews if c not in punctuation])
reviews = all_text.split('\n')

all_text = ' '.join(reviews)
words = all_text.split()


# Encoding the words

# Create dictionary that maps vocab words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

# Convert the reviews to integers, same shape as reviews list, but with integers
reviews_ints = []

for review in reviews:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])


# Encoding the labels

# Convert labels to 1s and 0s for 'positive' and 'negative'
labels = labels.split('\n')
labels = np.array([1 if each == 'positive' else 0 for each in labels])

# Filter out that review with 0 length
non_zero_idx = [i for i, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[i] for i in non_zero_idx]
labels = np.array([labels[i] for i in non_zero_idx])

seq_len = 200
features = np.zeros((len(reviews_ints), seq_len), dtype=int)

for i, review in enumerate(reviews_ints):
    features[i, -len(review):] = np.array(review[:seq_len])


# Create the training, validation, and test sets
split_frac = 0.8

split_1 = int(len(features) * 0.8)
train_x, val_x = features[:split_1], features[split_1:]
train_y, val_y = labels[:split_1], labels[split_1:]

split_2 = int(len(val_x) * 0.5)
val_x, test_x = val_x[:split_2], val_x[split_2:]
val_y, test_y = val_y[:split_2], val_y[split_2:]


# Build the Graph

n_words = len(vocab_to_int)

# Create the graph object
graph = tf.Graph()

# Add nodes to the graph
with graph.as_default():
    with tf.name_scope('inputs'):
        inputs_ = tf.placeholder(tf.int32, [None, None])

    with tf.name_scope('labels'):
        labels_ = tf.placeholder(tf.int32, [None, None])

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)


# Embedding

# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300

with graph.as_default():
    with tf.name_scope('embedding'):
        with tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        tf.summary.histogram('embedding', embedding)

        with tf.name_scope('embed'):
            embed = tf.nn.embedding_lookup(embedding, inputs_)


# LSTM cell

with graph.as_default():
    with tf.name_scope('LSTM'):
        def single_cell():
            # Basic LSTM cell
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            # Add dropout to the cell
            drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            return drop

        # Stack up multiple LSTM layers, for deep learning
        with tf.name_scope('lstm_cell'):
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(lstm_layers)],
                                               state_is_tuple=True)

        # Getting an initial state of all zeros
        with tf.name_scope('initial_state'):
            initial_state = cell.zero_state(batch_size, tf.float32)


# RNN forward pass

with graph.as_default():
    with tf.name_scope('forward_pass'):
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

# Output

with graph.as_default():
    with tf.name_scope('prediction'):
        with tf.name_scope('predictions'):
            predictions = tf.contrib.layers.fully_connected(rnn_outputs[:, -1], 1,
                                                            activation_fn=tf.sigmoid)
        tf.summary.histogram('predictions', predictions)

        with tf.name_scope('cost'):
            cost = tf.losses.mean_squared_error(labels_, predictions)
        tf.summary.scalar('cost', cost)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Validation accuracy

with graph.as_default():
    with tf.name_scope('validation'):
        with tf.name_scope('correct_pred'):
            correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy', accuracy)


# Batching

def get_batches(x, y, batch_size=100):

    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# Training

# Save
with graph.as_default():
    saver = tf.train.Saver(max_to_keep=100)

# Train
with tf.Session(graph=graph) as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summary_path + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(summary_path + '/validation')

    sess.run(tf.global_variables_initializer())

    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)

        for batch_i, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: keep_probability,
                    initial_state: state}
            train_summary, loss, state, _ = sess.run([merged, cost, final_state, optimizer],
                                                     feed_dict=feed)

            train_writer.add_summary(train_summary, iteration)

            if iteration % 10 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    val_summary, batch_acc, val_state = sess.run([merged, accuracy, final_state],
                                                                 feed_dict=feed)
                    val_acc.append(batch_acc)
                val_writer.add_summary(val_summary, iteration)

                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss),
                      "Val acc: {:.3f}".format(np.mean(val_acc)))

            iteration += 1

    saver.save(sess, "./checkpoints/sentiment.ckpt")
