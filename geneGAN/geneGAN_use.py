import sys, time, argparse
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm

_VALIDATION_RATIO = 0.1


class geneGAN_model(object):
    def __init__(self,
                 dataType='binary',
                 inputDim=615,
                 embeddingDim=128,
                 randomDim=128,
                 generatorDims=(128, 128),
                 discriminatorDims=(256, 128, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):
        self.inputDim = inputDim
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
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_' + str(i), shape=[tempDim, compressDim])
                b = tf.get_variable('aee_b_' + str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1

            i = 0
            for decompressDim in self.decompressDims[:-1]:

                W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, decompressDim])
                b = tf.get_variable('aed_b_' + str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_' + str(i)] = W
                decodeVariables['aed_b_' + str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_' + str(i), shape=[tempDim, self.decompressDims[-1]])
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

    def getDiscriminatorResults(self, x_input, x_mean, keepRate, reuse=False):

        batchSize = tf.shape(x_input)[0]
        inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_mean, 0), [batchSize]), (batchSize, self.inputDim))

        tempVec = tf.concat([x_input, inputMean, tf.zeros([batchSize, 1])], 1)

        tempDim = self.inputDim * 2 + 1
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat

    def getDiscriminatorResultsSingle(self, x_input, keepRate, reuse=False):

        batchSize = tf.shape(x_input)[0]

        tempVec = tf.concat([x_input, tf.zeros([batchSize, self.inputDim]), tf.ones([batchSize, 1])], 1)

        tempDim = self.inputDim * 2 + 1
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat

    def getDiscriminatorEmbeddingsSingle(self, x_input, keepRate, reuse=False):

        batchSize = tf.shape(x_input)[0]

        tempVec = tf.concat([x_input, tf.zeros([batchSize, self.inputDim]), tf.ones([batchSize, 1])], 1)
        embs = {}
        tempDim = self.inputDim * 2 + 1
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                embs[str(i)] = h
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
        return embs

    def buildGeneratorTest(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output

    def generateData(self, nSamples=100,
                     modelFile='model',
                     batchSize=100,
                     outFile='out',
                     random_input=None):
        tf.reset_default_graph()
        x_dummy = tf.placeholder('float', [None, self.inputDim])
        _, decodeVariables = self.buildAutoencoder(x_dummy)
        x_random = tf.placeholder('float', [None, self.randomDim])
        bn_train = tf.placeholder('bool')

        x_emb = self.buildGeneratorTest(x_random, bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        #np.random.seed(1234)
        saver = tf.train.Saver()
        outputVec = []

        with tf.Session() as sess:
            saver.restore(sess, modelFile)

            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in range(nBatches):
                if random_input is not None and tuple(np.shape(random_input))==(batchSize*nBatches, self.randomDim):
                    randomX = random_input[i*batchSize:(i+1)*batchSize,:]
                else:
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = sess.run(x_reconst, feed_dict={x_random:randomX, bn_train:False})
                outputVec.extend(output)

        outputMat = np.array(outputVec)
        #np.save(outFile, outputMat)
        return outputMat
    
    def discriminateData(self, data, data_mean, modelFile, batchSize=1000, keepRate=1.0, single=False):
        tf.reset_default_graph()
        input_data = tf.placeholder('float', [None, self.inputDim])
        if single:
            y_hat = self.getDiscriminatorResultsSingle(input_data, keepRate)
        else:
            input_mean = tf.placeholder('float', [None, self.inputDim])
            y_hat = self.getDiscriminatorResults(input_data, input_mean, keepRate)
        saver = tf.train.Saver()
        results = []

        nBatches = int(np.ceil(np.shape(data)[0]/batchSize))

        with tf.Session() as sess:
            saver.restore(sess, modelFile)
            for i in range(nBatches):
                if single:
                    result = sess.run(y_hat, feed_dict={input_data: data[(i * batchSize):(i + 1) * batchSize, :]})
                else:
                    result = sess.run(y_hat, feed_dict={input_data: data[(i*batchSize):(i+1)*batchSize,:],
                                                        input_mean: data_mean[(i*batchSize):(i+1)*batchSize,:]})

                results.extend(result)
        return results

    def getDiscEmbeddings(self, data, modelFile, batchSize=1000, keepRate=1.0):
        tf.reset_default_graph()
        input_data = tf.placeholder('float', [None, self.inputDim])

        embeddings = self.getDiscriminatorEmbeddingsSingle(input_data, keepRate)
        saver = tf.train.Saver()
        results={}
        for key in embeddings:
            results[key] = []

        nBatches = int(np.ceil(np.shape(data)[0] / batchSize))

        with tf.Session() as sess:
            saver.restore(sess, modelFile)
            for i in range(nBatches):

                result = sess.run(embeddings, feed_dict={input_data: data[(i * batchSize):(i + 1) * batchSize, :]})

                for key in result:
                    results[key].extend(result[key])
        return results