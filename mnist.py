from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import heapq
import random


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv))
        self.logits = tf.nn.softmax(y_conv)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_eval_preds(xes, yes, model, legal_indices, batch_size=1024):
    choices = []
    for i in range(0, len(xes), batch_size):
        preds = \
            model.logits.eval(
                feed_dict={model.x: xes[i: i + batch_size], model.y_: yes[i: i + batch_size],
                           model.keep_prob: 1.0})
        entropies = - np.sum(preds * np.log(preds), axis=1)
        for offset, x in enumerate(entropies):
            index = offset + i
            if index in legal_indices:
                heapq.heappush(choices, (x, index))
            if len(choices) > 10:
                heapq.heappop(choices)
    chosen_indices = []
    for _, index in choices:
        legal_indices.remove(index)
        chosen_indices.append(index)
    return chosen_indices


def main():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    xes = mnist.train.images
    yes = mnist.train.labels
    remaining_indices = set(list(range(len(xes))))
    print(len(mnist.train.images), len(mnist.validation.images), len(mnist.test.images))
    print('-' * 100)
    model = Model()
    all_train_indices = []
    batch_size = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            batch_indices = get_eval_preds(xes, yes, model, remaining_indices)
            all_train_indices.extend(batch_indices)

            for _ in range(20000):
                total_loss = 0
                random.shuffle(all_train_indices)
                cur_index = 0
                while cur_index < len(all_train_indices):
                    batch_indices = all_train_indices[cur_index: cur_index + batch_size]
                    cur_index += batch_size
                    _, loss = sess.run([model.train_step, model.cross_entropy],
                                       feed_dict={
                                           model.x: [xes[i] for i in batch_indices],
                                           model.y_: [yes[i] for i in batch_indices],
                                           model.keep_prob: 0.5
                                       })
                    total_loss += loss
                if total_loss < 0.00001:
                    print('total loss', total_loss)
                    break
            print(len(all_train_indices))
            print('test accuracy %g' % model.accuracy.eval(
                feed_dict={
                    model.x: mnist.test.images,
                    model.y_: mnist.test.labels,
                    model.keep_prob: 1.0
                }
            ))


if __name__ == '__main__':
    main()
