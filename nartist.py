"""
Neural Network to generate images
"""

import numpy as np
import tensorflow as tf


class NArtist:
    def __init__(self, width, height, channels,
                 learn_rate, batch_size,
                 title_length_max, clarity):
        """
        Initialize NArtist
        :param width: Width of image in pixels
        :param height: Height of image in pixels
        :param channels: Number of color channels
        :param learn_rate: Learning rate of network
        :param title_length_max: Maximum length of a title
        :param clarity: Clarity of image (Number of binary classifiers)
        """

        # Check arguments
        assert batch_size > 0  # must be positive
        assert clarity > 0  # must be positive
        assert channels > 0  # must be positive
        assert height > 0  # must be positive
        assert learn_rate > 0  # must be positive
        assert title_length_max > 0  # must be positive
        assert width > 0  # must be positive

        # Assign arguments
        self.batch_size = batch_size  # set batch size
        self.clarity = clarity  # set clarity
        self.channels = channels  # set channels
        self.height = height  # set height
        self.learn_rate = learn_rate  # set learning rate
        self.title_length_max = title_length_max  # set title length max
        self.width = width  # set width

        # Create network
        self.tens_type = tf.float32  # set tensor type
        self.inputs = tf.placeholder(self.tens_type, [None, title_length_max], name="inputs")  # create input numbers
        self.labels = tf.placeholder(self.tens_type, [None, height, width, channels, clarity],
                                     name="labels")  # create logits

        self.h1 = tf.layers.dense(self.inputs, 10, name="h1",
                                  kernel_initializer=tf.truncated_normal_initializer(
                                      stddev=1.0/np.sqrt(100)/127))  # hidden layer 1
        self.logits = tf.layers.dense(self.h1, height * width * channels * clarity)  # create logits
        self.logits = tf.reshape(self.logits, [-1, height, width, channels, clarity], name="logits")  # reshape

        # Cost, optimizer, accuracy
        self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                                           logits=self.logits),
                                   axis=0, name="cost")  # cost over batches
        self.mean_cost = tf.reduce_mean(self.cost)  # mean cost
        self.accur = 1.0 - self.mean_cost  # get accuracy
        self.optim = tf.train.RMSPropOptimizer(learn_rate).minimize(self.cost)  # optimize cost
        self.prediction = tf.argmax(self.logits, -1, name="prediction")  # get predicted image

        # Create session
        self.sess = tf.Session()  # create session
        self.init_op = tf.global_variables_initializer()  # create initializer
        self.sess.run(self.init_op)  # initialize network

    def con_image(self, logits):
        """
        Convert logits to image
        :param logits: Logits of network
        :return: Image
        """
        return np.multiply(np.divide(logits, self.clarity), 255)  # return

    def con_image_vect(self, images):
        """
        Convert Images to vector
        :param images: Images to convert
        :return: Vector array
        """
        labels = np.zeros([len(images), self.height, self.width, self.channels, self.clarity])  # declare labels
        vect = np.clip(np.floor(np.multiply(np.divide(images, 255), self.clarity)),
                       0, self.clarity-1).astype(int)  # divide and clamp to clarity

        for image in range(len(images)):
            for col in range(self.height):
                for row in range(self.width):
                    for channel in range(self.channels):
                        labels[image][col][row][channel][vect[image][col][row][channel]] = 1.0  # set label
        return labels  # return labels

    def con_title_vect(self, titles):
        """
        Convert title to vector
        :param titles: Titles to convert
        :return: Vector array
        """
        return [[ord(char) for char in title[:self.title_length_max].ljust(100)] for title in titles]  # Convert

    def predict(self, title):
        """
        Predict the image from the title
        :param title: Title of the image
        :return: Image
        """
        inputs = self.con_title_vect([title])  # convert
        prediction = self.sess.run([self.prediction], feed_dict={self.inputs: inputs})  # predict image
        images = self.con_image(prediction[0])  # return converted image
        return images[0]  # return prediction

    def train(self, titles, images, epochs, progress_every=1):
        """
        Train network
        :param titles: List of names of images
        :param images:  List of images to train for
        :param epochs: Epochs on data
        :param progress_every: Write progress every x epochs
        :return: None
        """
        # Check arguments
        assert epochs > 0  # must be positive

        inputs = self.con_title_vect(titles)  # convert titles
        labels = self.con_image_vect(images)  # convert images

        batches = np.floor(len(titles) / self.batch_size).astype(int)

        # train
        for i_epoch in range(epochs):  # for epochs
            if i_epoch % progress_every == 0:  # if write progress
                epoch_accur, epoch_cost = 0, 0  # rest accuracy and cost

                for i_batch in range(batches):  # for batches
                    batch_input = inputs[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]  # get input
                    batch_label = labels[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]  # get label

                    batch_accur, batch_cost, _ = self.sess.run([self.accur, self.mean_cost, self.optim],
                                                               {self.inputs: batch_input,
                                                                self.labels: batch_label})  # train

                    epoch_accur += batch_accur / batches  # increment
                    epoch_cost += batch_cost / batches  # increment

                print("Epoch {0}: Accuracy: {1}, Cost: {2}".format(i_epoch,
                                                                   epoch_accur, epoch_cost))  # write results
            else:
                for i_batch in range(batches):  # for batches
                    batch_input = inputs[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]  # get input
                    batch_label = labels[i_batch * self.batch_size:(i_batch + 1) * self.batch_size]  # get label

                    self.sess.run([self.optim], {self.inputs: batch_input, self.labels: batch_label})  # train



