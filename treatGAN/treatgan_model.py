import sys, time, argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm

_VALIDATION_RATIO = 0.1


class treatgan_load(object):

    def __init__(self,
                 modelPath,
                 dataType='binary',
                 inputDim=615,
                 condDim=200,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):

        self.modelPath = modelPath
        self.inputDim = inputDim
        self.condDim = condDim
        self.embeddingDim = embeddingDim
        self.generatorDims = list(generatorDims) + [embeddingDim]
        self.randomDim = randomDim
        self.dataType = dataType

        if dataType == 'binary':
            self.aeActivation = tf.nn.tanh
        else:
            self.aeActivation = tf.nn.relu

        self.generatorActivation = tf.nn.relu
        self.discriminatorActivation = tf.nn.relu
        self.discriminatorDims = discriminatorDims
        self.compressDims = list(compressDims) + [embeddingDim]
        self.decompressDims = list(decompressDims) + [inputDim]

        self.bnDecay = bnDecay
        self.l2scale = l2scale

    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.variable_scope('autoencoder'):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_' + str(i), shape=[tempDim, compressDim],
                                    regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('aee_b_' + str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1

            i = 0
            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, decompressDim],
                                    regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_' + str(i)] = W
                decodeVariables['aed_b_' + str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, self.decompressDims[-1]],
                                regularizer=l2_regularizer(self.l2scale))
            b = tf.get_variable('aed_b_' + str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_' + str(i)] = W
            decodeVariables['aed_b_' + str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean(-tf.reduce_sum(
                    x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, W), b))
                loss = tf.reduce_mean((x_input - x_reconst) ** 2)

        return loss, decodeVariables

    def getDiscriminatorResults(self, x_input, x_cond, keepRate=1.0, reuse=False):

        tempVec = tf.concat([x_cond, x_input], 1)
        tempDim = self.inputDim + self.condDim

        with tf.variable_scope('discriminator', reuse=reuse):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, discDim], regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('b_' + str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec, W), b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1], regularizer=l2_regularizer(self.l2scale))
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat

    def buildGeneratorTest(self, x_input, x_cond, bn_train):
        tempVec = tf.concat([x_input, x_cond],1)
        tempDim = self.randomDim + self.condDim
        with tf.variable_scope('generator'):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[tempDim, genDim], regularizer=l2_regularizer(self.l2scale))
                h = tf.matmul(tempVec, W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None,
                                trainable=False)
                h3 = self.generatorActivation(h2)

                tempVecUse, rest = tf.split(tempVec, [genDim, tempDim - genDim], axis=1)
                tempVec = h3 + tempVecUse
                tempDim = genDim
            W = tf.get_variable('W' + str(i), shape=[tempDim, self.generatorDims[-1]],
                                regularizer=l2_regularizer(self.l2scale))
            h = tf.matmul(tempVec, W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None,
                            trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)
            output = h3 + tempVec
        return output

    def generateData(self, x_cond_data):

        tf.reset_default_graph()
        x_dummy = tf.placeholder('float', [None, self.inputDim])
        x_cond = tf.placeholder('float', [None, self.condDim])
        x_random = tf.placeholder('float', [None, self.randomDim])
        bn_train = tf.placeholder('bool')

        _, decodeVariables = self.buildAutoencoder(x_dummy)

        x_emb = self.buildGeneratorTest(x_random, x_cond, bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        saver = tf.train.Saver()
        outputVec = []

        with tf.Session() as sess:
            saver.restore(sess, self.modelPath)

            randomX = np.random.normal(size=(np.shape(x_cond_data)[0], self.randomDim))
            output = sess.run(x_reconst, feed_dict={x_random:randomX, x_cond: x_cond_data, bn_train:False})
            outputVec.extend(output)

        outputMat = np.array(outputVec)
        return outputMat
    
    def discriminateData(self, data, x_cond_data):
        tf.reset_default_graph()
        input_data = tf.placeholder('float', [None, self.inputDim])
        x_cond = tf.placeholder('float', [None, self.condDim])
        y_hat = self.getDiscriminatorResults(input_data, x_cond)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.modelPath)
            disc_value = sess.run(y_hat, feed_dict={input_data: data, x_cond: x_cond_data})

        return np.array(disc_value)