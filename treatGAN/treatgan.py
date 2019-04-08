# noinspection PyInterpreter
import sys, time, argparse
import tensorflow as tf
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm
import matplotlib.pyplot as plt
from collections import Counter

_VALIDATION_RATIO = 0.1


class treatGAN(object):

    def __init__(self,
                 dataType='binary',
                 inputDim=121,
                 condDim=200,
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
        self.condDim = condDim
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

        print()
        print('configuration:')
        print('---------------------------------')
        print('input dim:\t', self.inputDim)
        print('cond dim:\t', self.condDim)
        print('random dim:\t', self.randomDim)
        print('embedding:\t', self.embeddingDim)
        print('compressor:\t', self.compressDims)
        print('decompressor:\t', self.decompressDims)
        print('generator:\t', self.generatorDims)
        print('discriminator:\t', self.discriminatorDims)
        print('---------------------------------')
        print()


    def loadData(self, dataPath=''):
        data = np.load(dataPath)

        if self.dataType == 'binary':
            data = np.clip(data, 0, 1)

        indices = list(range(0,np.shape(data)[0]))

        train_ind, valid_ind =  train_test_split(indices, test_size=_VALIDATION_RATIO, random_state=0)

        trainX, validX = data[train_ind,:], data[valid_ind,:]
        return trainX, validX, train_ind, valid_ind

    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.variable_scope('autoencoder'):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_'+str(i), shape=[tempDim, compressDim], regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('aee_b_'+str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1
    
            i = 0
            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, decompressDim], regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('aed_b_'+str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_'+str(i)] = W
                decodeVariables['aed_b_'+str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, self.decompressDims[-1]], regularizer=l2_regularizer(self.l2scale))
            b = tf.get_variable('aed_b_'+str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_'+str(i)] = W
            decodeVariables['aed_b_'+str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean(-tf.reduce_sum(x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean((x_input - x_reconst)**2)
            
        return loss, decodeVariables

    def buildGenerator(self, x_input, x_cond, bn_train):
        tempVec = tf.concat([x_input, x_cond],1)
        tempDim = self.randomDim+self.condDim
        with tf.variable_scope('generator'):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim], regularizer=l2_regularizer(self.l2scale))
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVecUse, rest =  tf.split(tempVec, [genDim, tempDim - genDim],axis=1)
                tempVec = h3 + tempVecUse
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]], regularizer=l2_regularizer(self.l2scale))
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def buildGeneratorTest(self, x_input, x_cond, bn_train):
        tempVec = tf.concat([x_input, x_cond])
        tempDim = self.randomDim + self.condDim
        with tf.variable_scope('generator'):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim], regularizer=l2_regularizer(self.l2scale))
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)

                tempVecUse, rest = tf.split(tempVec, [genDim, tempDim - genDim], axis=1)
                tempVec = h3 + tempVecUse
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]], regularizer=l2_regularizer(self.l2scale))
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)
            output = h3 + tempVec
        return output
    
    def getDiscriminatorResults(self, x_input,x_cond, keepRate, single_mode, reuse=False):

        tempVec = tf.concat([x_cond, x_input],1)
        tempDim = self.inputDim + self.condDim

        with tf.variable_scope('discriminator', reuse=reuse):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim], regularizer=l2_regularizer(self.l2scale))
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1], regularizer=l2_regularizer(self.l2scale))
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat
    
    def buildDiscriminator(self, x_real, x_fake, x_cond, x_random, keepRate, decodeVariables, bn_train, single_mode):

        const_true = tf.constant(True, dtype=tf.bool)
        const_false = tf.constant(False, dtype=tf.bool)
        #Discriminate for real samples

        y_hat_random_single = self.getDiscriminatorResults(x_random, x_cond, keepRate, const_true, reuse=False)
        y_hat_real_single = self.getDiscriminatorResults(x_real, x_cond, keepRate, const_true, reuse=True)
        y_hat_real_batch = self.getDiscriminatorResults(x_real, x_cond,keepRate, const_false, reuse=True)
        #Decompress, then discriminate for real samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        y_hat_fake_single = self.getDiscriminatorResults(x_decoded, x_cond, keepRate, const_true, reuse=True)
        y_hat_fake_batch = self.getDiscriminatorResults(x_decoded, x_cond, keepRate, const_false, reuse=True)

        loss_d_single = -tf.reduce_mean(tf.log(y_hat_real_single + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake_single + 1e-12))
        loss_d_batch  = -tf.reduce_mean(tf.log(y_hat_real_batch + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake_batch + 1e-12))

        loss_g_single = -tf.reduce_mean(tf.log(y_hat_fake_single + 1e-12))
        loss_g_batch = -tf.reduce_mean(tf.log(y_hat_fake_batch + 1e-12))

        #loss_g_pre = tf.losses.log_loss(x_real, x_decoded)
        #loss_g_pre = tf.contrib.losses.mean_squared_error(x_real, x_decoded)
        loss_g_pre =  tf.reduce_mean((x_real - x_decoded) ** 2)
        loss_d_pre = -tf.reduce_mean(tf.log(y_hat_real_single + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_random_single + 1e-12))

        beta = 0.1
        loss_d = loss_d_batch
        loss_g = loss_g_batch

        loss_d_comb = loss_d_single
        loss_g_comb = loss_g_single

        loss_pre = [loss_g_pre, loss_d_pre]
        losses = [loss_d, loss_g, loss_d_comb, loss_g_comb]
        y_hats = [y_hat_real_single, y_hat_real_batch,y_hat_fake_single, y_hat_fake_batch, y_hat_random_single]

        return loss_pre, losses, y_hats, x_decoded

    def print2file(self, buf, outFile):
        outfd = open(outFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()
    
    def generateData(self,
                     nSamples=100,
                     modelFile='model',
                     batchSize=100,
                     outFile='out'):
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

        np.random.seed(1234)
        saver = tf.train.Saver()
        outputVec = []
        burn_in = 1000
        with tf.Session() as sess:
            saver.restore(sess, modelFile)
            print('burning in')
            for i in range(burn_in):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                _ = sess.run(x_reconst, feed_dict={x_random:randomX, bn_train:True})

            print('generating')
            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in range(nBatches):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = sess.run(x_reconst, feed_dict={x_random:randomX, bn_train:False})
                outputVec.extend(output)

        outputMat = np.array(outputVec)
        np.save(outFile, outputMat)
    
    def calculateDiscAuc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc
    
    def calculateDiscAccuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real: 
            if pred > 0.5: hit += 1
        for pred in preds_fake: 
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc

    def sortListAndLabel(self, label_list, unordered_list):
        ordered_list, ordered_label = (list(t) for t in zip(*sorted(zip(unordered_list, label_list))))
        return ordered_label, ordered_list

    def calculateDiscQuality(self, preds_real, preds_fake, boundary=None):

        if boundary:
            best_success_real = np.sum(1. for y in preds_real if y >= boundary) / len(preds_real)
            best_success_fake = np.sum(1. for y in preds_fake if y >= boundary) / len(preds_fake)
            best_score = np.sqrt((best_success_real - 1) ** 2 + best_success_fake ** 2)
            best_bound = boundary

            unscaled_success_real = np.sum(1. for y in preds_real if y >= 0.5) / len(preds_real)
            unscaled_success_fake = np.sum(1. for y in preds_fake if y >= 0.5) / len(preds_fake)
            unscaled_score = np.sqrt((unscaled_success_real - 1) ** 2 + unscaled_success_fake ** 2)

        else:
            tmp_score = []
            tmp_s_in = []
            tmp_s_out = []
            tmp_ind  =[]

            for i in range(9):
                bound = (i + 1) / 10
                success_real = np.sum(1. for y in preds_real if y >= bound) / len(preds_real)
                success_fake = np.sum(1. for y in preds_fake if y >= bound) / len(preds_fake)

                tmp_score.append(np.sqrt((success_real - 1) ** 2 + success_fake ** 2))
                tmp_s_in.append(success_real)
                tmp_s_out.append(success_fake)
                tmp_ind.append(i)

                if i == 4:
                    unscaled_success_real = success_real
                    unscaled_success_fake = success_fake
                    unscaled_score = tmp_score[i]

            sorted_ind, sorted_score = self.sortListAndLabel(tmp_ind, tmp_score)

            best_bound = (sorted_ind[0] + 1) / 10
            best_success_real = tmp_s_in[sorted_ind[0]]
            best_success_fake = tmp_s_out[sorted_ind[0]]
            best_score = sorted_score[0]

        return unscaled_success_real, unscaled_success_fake, unscaled_score,\
               best_success_real, best_success_fake, best_score, best_bound

    def countValidPairs(self, genData):

        tupleDict = {}
        nbr_valid = 0
        n_samples = np.shape(genData)[0]
        roundedData = np.round(genData)

        for i in range(n_samples):
            ind = np.nonzero(roundedData[i, :])[0]
            if len(ind) == 2:

                nbr_valid += 1
                dataTuple = tuple((ind[0], ind[1]))

                if dataTuple in tupleDict:
                    tupleDict[dataTuple]+=1
                else:
                    tupleDict[dataTuple]=1

        return nbr_valid, len(tupleDict), list(tupleDict.keys())

    def conditionalOutputAnalysis(self, generated_data, expected_data):
        data_dim = expected_data.shape[1]
        diffs_to_round = []
        diffs_to_exp = []
        diffs_nonzero = []
        round_sums = []
        overlap = []
        correct_share = []
        wrong_digits_counter = Counter()
        for i in range(np.shape(generated_data)[0]):
            output_vec = generated_data[i,:]
            rounded_vec = np.round(output_vec)
            exp_vec = expected_data[i,:]

            gen_nonzero = set(np.nonzero(rounded_vec)[0])
            exp_nonzero = set(np.nonzero(exp_vec)[0])



            try:
                overlap.append(len(gen_nonzero & exp_nonzero)/len(gen_nonzero | exp_nonzero))
            except:
                if len(gen_nonzero)==0 and len(exp_nonzero)==0:
                    overlap.append(1)
                else:
                    overlap.append(0)


            correct_share.append((data_dim-len(gen_nonzero ^ exp_nonzero))/data_dim)


            wrong_digits_counter.update(gen_nonzero ^ exp_nonzero)

            diff_to_round = np.sum(np.abs(output_vec-rounded_vec))
            diff_to_exp = np.sum(np.abs(rounded_vec-exp_vec))
            diff_nonzero = np.sum(rounded_vec)-np.sum(exp_vec)
            round_sum = np.sum(rounded_vec)

            diffs_to_round.append(diff_to_round)
            diffs_to_exp.append(diff_to_exp)
            diffs_nonzero.append(diff_nonzero)
            round_sums.append(round_sum)

        return diffs_to_round, diffs_to_exp, diffs_nonzero, round_sums, overlap, wrong_digits_counter, correct_share

    def importTrainAndValidData(self, modelPath, dataPath, outPath):
        if modelPath=='':

            trainX, validX, train_ind, valid_ind = self.loadData(dataPath+'_treat.npy')
            data_demo = np.load(dataPath+'_demo.npy')
            trainCond, validCond = data_demo[train_ind, :], data_demo[valid_ind, :]

        else:
            data = np.load(dataPath+'_treat.npy')
            train_ind = np.load(os.path.dirname(modelPath)+'/train_ind.npy')
            valid_ind = np.load(os.path.dirname(modelPath) + '/valid_ind.npy')
            trainX, validX = data[train_ind, :], data[valid_ind, :]

            data_demo = np.load(dataPath + '_demo.npy')
            trainCond, validCond = data_demo[train_ind, :], data_demo[valid_ind, :]

            np.save(os.path.dirname(outPath) + '/train_ind.npy', train_ind)
            np.save(os.path.dirname(outPath) + '/valid_ind.npy', valid_ind)

        return trainX, validX, trainCond, validCond, train_ind, valid_ind

    def pretrainAutoencoder(self,sess, pretrainEpochs, pretrainBatchSize, optimize_ae, loss_ae, x_raw, training=10000, valid=1000):

        '''pretraining of the autoencoder with random n-hot vectors

        :param sess: tensorflow session
        :param pretrainEpochs: number of epochs to train autoencoder
        :param pretrainBatchSize: batch size for autoencoder training
        :param optimize_ae: optimizer used for training
        :param loss_ae: tf variable for autoencoder loss
        :param x_raw: tf variable for autoencoder input
        :param training: training number of samples. 10000 by default
        :param valid: validation number of samples.  1000 by default
        :return: list with training results (losses)
        '''

        ae_training_status = []

        nTrainBatches = int(np.ceil(float(training)) / float(pretrainBatchSize))
        nValidBatches = int(np.ceil(float(valid)) / float(pretrainBatchSize))

        for epoch in range(pretrainEpochs):
            t1 = time.time()
            trainLossVec = []
            for i in range(nTrainBatches):
                batchX = []
                for j in range(pretrainBatchSize):
                    vec = np.zeros(self.inputDim)
                    ind_to_one = random.sample(range(0, self.inputDim), random.randint(0, self.inputDim))
                    vec[ind_to_one] = 1
                    batchX.append(vec)
                batchX = np.array(batchX)
                _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw: batchX})
                trainLossVec.append(loss)

            validLossVec = []
            for i in range(nValidBatches):
                batchX = []
                for j in range(pretrainBatchSize):
                    vec = np.zeros(self.inputDim)
                    ind_to_one = random.sample(range(0, self.inputDim), random.randint(0, self.inputDim))
                    vec[ind_to_one] = 1
                    batchX.append(vec)
                batchX = np.array(batchX)
                loss = sess.run(loss_ae, feed_dict={x_raw: batchX})
                validLossVec.append(loss)
            validReverseLoss = 0.
            buf = 'pretrain_epoch:%d, trainLoss:%f, validLoss:%f' % (
            epoch, np.mean(trainLossVec), np.mean(validLossVec))
            print(buf)
            t2 = time.time()
            delta_t = t2 - t1
            t1 = t2
            ae_training_status.append([epoch, delta_t, np.mean(trainLossVec), np.mean(validLossVec)])
        return ae_training_status

    def pretrainGANstep(self, sess, trainX, trainCond, batchSize, optimize_d_pre, optimize_g_pre, loss_d_pre, loss_g_pre,
                        x_raw, x_random, x_pre_rand, x_cond, bn_train, keep_prob, keepProb):
        idx = np.arange(trainX.shape[0])
        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))

        d_loss_vec = []
        g_loss_vec = []

        for _ in range(nBatches):
            batchIdx = np.random.choice(idx, size=batchSize, replace=False)
            batchX = trainX[batchIdx]
            batchC = trainCond[batchIdx]

            batchRandX = []
            for k in range(batchSize):
                vec = np.zeros(self.inputDim)
                ind_to_one = random.sample(range(0, np.shape(batchX)[1]), np.sum(batchX[k, :]))
                vec[ind_to_one] = 1
                batchRandX.append(vec)
            batchRandX = np.array(batchRandX)

            randomX = np.zeros((batchSize, self.randomDim))
            _, discLoss = sess.run([optimize_d_pre, loss_d_pre],
                                   feed_dict={x_raw: batchX, x_random: randomX, x_pre_rand: batchRandX,
                                              x_cond: batchC, keep_prob: keepProb,
                                              bn_train: False})
            d_loss_vec.append(discLoss)

            randomX = np.random.normal(size=(batchSize, self.randomDim))
            _, generatorLoss = sess.run([optimize_g_pre, loss_g_pre],
                                        feed_dict={x_raw: batchX, x_random: randomX,
                                                   x_cond: batchC, keep_prob: 1.0,
                                                   bn_train: True})
            g_loss_vec.append(generatorLoss)

        return np.mean(d_loss_vec), np.mean(g_loss_vec)

    def pretrainGANvalid(self, sess, x, cond ,vBatches, batchSize, loss_g_pre,
                        x_raw, x_random, x_pre_rand,x_cond, bn_train, keep_prob,
                         y_hat_real_single, y_hat_random_single):

        idx = np.arange(x.shape[0])
        g_loss_vec = []
        disc_accuracy = []
        batchSize = np.min([batchSize, x.shape[0]])
        for i in range(vBatches):
            batchIdx = np.random.choice(idx, size=batchSize, replace=False)
            batchX = x[batchIdx]
            batchC = cond[batchIdx]

            batchRandX = []
            for k in range(batchSize):
                vec = np.zeros(self.inputDim)
                ind_to_one = random.sample(range(0, np.shape(batchX)[1]), np.sum(batchX[k, :]))
                vec[ind_to_one] = 1
                batchRandX.append(vec)
            batchRandX = np.array(batchRandX)

            randomX = np.zeros((batchSize, self.randomDim))
            y_hat_real, y_hat_rand = sess.run([y_hat_real_single, y_hat_random_single],
                                              feed_dict={x_raw: batchX, x_random: randomX, x_pre_rand: batchRandX,
                                                         x_cond: batchC, keep_prob: 1.0, bn_train: False})

            d_acc = self.calculateDiscAccuracy(y_hat_real, y_hat_rand)
            disc_accuracy.append(d_acc)

        return np.mean(disc_accuracy)

    def pretrainGAN(self, sess, pretrainEpochs, batchSize, trainX, trainCond, validX, validCond, optimize_d_pre,
                    loss_d_pre, optimize_g_pre,loss_g_pre, x_raw, x_random, x_pre_rand, x_cond,x_synth, keep_prob, keepProb,
                    bn_train,y_hats, logFile):

        pre_train_status = []
        pre_gen_status = []
        pre_counter_valid = Counter()
        pre_counter_train = Counter()

        for epoch in range(pretrainEpochs):
            t1 = time.time()

            '''
            pretrain GAN for one epoch
            '''
            d_loss_vec, g_loss_vec = self.pretrainGANstep(sess, trainX, trainCond, batchSize, optimize_d_pre, optimize_g_pre,
                                                     loss_d_pre, loss_g_pre, x_raw, x_random, x_pre_rand,x_cond,
                                                     bn_train, keep_prob, keepProb)


            '''
            evalute training step
            '''

            validation_results = self.validStep( sess, trainX, validX, trainCond, validCond, batchSize, x_raw, x_cond,
                                                 x_random, y_hats, keep_prob, bn_train)

            gen_result_vec, _, _, counter_train, counter_valid = self.validGeneratorOutput(sess, 1, 2, 1,validX, trainX, trainCond, validCond,
                                                        x_raw, x_cond, x_random, x_synth, keep_prob, bn_train,
                                                        loss_g_pre, [], [])

            if (pretrainEpochs-epoch)<100:
                pre_counter_valid.update(counter_valid)
                pre_counter_train.update(counter_train)

            delta_t = time.time()-t1

            buf = 'pre_epoch:%05d, time:%.2f d_loss:%.2f, g_loss:%.2f, acc_train:%.2f, acc_valid:%.2f, g_loss_pre_train:%.2f, g_loss_pre_valid:%.2f, overlap_train:%.2f, overlap_valid::%.2f' \
                  % (epoch, delta_t, np.mean(d_loss_vec), np.mean(g_loss_vec), validation_results[6], validation_results[2],
                     gen_result_vec[8], gen_result_vec[9], gen_result_vec[10], gen_result_vec[11])

            print(buf)
            self.print2file(buf, logFile)
            status = [epoch, delta_t, np.mean(d_loss_vec), np.mean(g_loss_vec)]
            status.extend(validation_results)
            pre_train_status.append(status)
            pre_gen_status.append(gen_result_vec)

        return pre_train_status, pre_gen_status, pre_counter_train, pre_counter_valid

    def trainStep(self, sess, batchSize, discriminatorTrainPeriod, generatorTrainPeriod, trainX, trainCond,
                  optimize_d_after,optimize_g_after,loss_d_comb,loss_g_comb, x_raw, x_random, x_cond, keep_prob, keepProb,
                  bn_train):

        idx = np.arange(trainX.shape[0])
        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        np.arange(trainX.shape[0])

        d_loss_vec = []
        g_loss_vec = []

        for i in range(nBatches):
            for _ in range(discriminatorTrainPeriod):
                batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                batchX = trainX[batchIdx]
                batchC = trainCond[batchIdx]
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                _, discLoss = sess.run([optimize_d_after, loss_d_comb],
                                       feed_dict={x_raw: batchX, x_random: randomX,
                                                  x_cond: batchC, keep_prob: keepProb,
                                                  bn_train: False})
                d_loss_vec.append(discLoss)

            for _ in range(generatorTrainPeriod):
                batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                batchX = trainX[batchIdx]
                batchC = trainCond[batchIdx]
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                _, generatorLoss = sess.run([optimize_g_after, loss_g_comb],
                                            feed_dict={x_raw: batchX, x_random: randomX,
                                                       x_cond: batchC, keep_prob: 1.0,
                                                       bn_train: True})
                g_loss_vec.append(generatorLoss)

        return d_loss_vec, g_loss_vec

    def validStep(self, sess, trainX, validX, trainCond, validCond, batchSize,
                  x_raw, x_cond, x_random, y_hats, keep_prob, bn_train):

        idx_valid = np.arange(len(validX))
        idx = np.arange(trainX.shape[0])

        y_hat_real_single, y_hat_real_batch, y_hat_fake_single, y_hat_fake_batch, y_hat_random_single = y_hats


        nValidBatches = int(np.ceil(float(len(idx_valid)) / float(batchSize)))
        validAccVec = []
        validAucVec = []
        validAccVec_single = []
        validAucVec_single = []

        trainAccVec = []
        trainAucVec = []
        trainAccVec_single = []
        trainAucVec_single = []

        for i in range(nValidBatches):
            ### validate with validation data ###
            vBatchSize = min(batchSize, len(idx_valid))
            batchIdx_valid = np.random.choice(idx_valid, size=vBatchSize, replace=False)
            batchX_valid = validX[batchIdx_valid]
            batchC_valid = validCond[batchIdx_valid]
            randomX = np.random.normal(size=(vBatchSize, self.randomDim))

            preds  = sess.run([y_hat_real_single, y_hat_real_batch, y_hat_fake_single, y_hat_fake_batch],
                              feed_dict={x_raw: batchX_valid,x_cond: batchC_valid,x_random: randomX,keep_prob: 1.0,bn_train: False})

            preds_real_single, preds_real_batch, preds_fake_single, preds_fake_batch = preds

            validAcc = self.calculateDiscAccuracy(preds_real_batch, preds_fake_batch)
            validAuc = self.calculateDiscAuc(preds_real_batch, preds_fake_batch)
            validAccVec.append(validAcc)
            validAucVec.append(validAuc)

            validAcc = self.calculateDiscAccuracy(preds_real_single, preds_fake_single)
            validAuc = self.calculateDiscAuc(preds_real_single, preds_fake_single)
            validAccVec_single.append(validAcc)
            validAucVec_single.append(validAuc)

            ### validate with training data ###
            vBatchSize = batchSize
            batchIdx_train = np.random.choice(idx, size=vBatchSize, replace=False)
            batchX_train = trainX[batchIdx_train]
            batchC_train = trainCond[batchIdx_train]
            randomX = np.random.normal(size=(vBatchSize, self.randomDim))

            preds = sess.run([y_hat_real_single, y_hat_real_batch, y_hat_fake_single, y_hat_fake_batch],
                             feed_dict={x_raw: batchX_train,x_cond: batchC_train,x_random: randomX,keep_prob: 1.0, bn_train: False})

            preds_real_single, preds_real_batch, preds_fake_single, preds_fake_batch = preds

            validAcc = self.calculateDiscAccuracy(preds_real_batch, preds_fake_batch)
            validAuc = self.calculateDiscAuc(preds_real_batch, preds_fake_batch)
            trainAccVec.append(validAcc)
            trainAucVec.append(validAuc)

            validAcc = self.calculateDiscAccuracy(preds_real_single, preds_fake_single)
            validAuc = self.calculateDiscAuc(preds_real_single, preds_fake_single)
            trainAccVec_single.append(validAcc)
            trainAucVec_single.append(validAuc)


        training_list = [np.mean(validAccVec),
                         np.mean(validAucVec),
                         np.mean(validAccVec_single),
                         np.mean(validAucVec_single),
                         np.mean(trainAccVec),
                         np.mean(trainAucVec),
                         np.mean(trainAccVec_single),
                         np.mean(trainAucVec_single)]

        return training_list


    def validGeneratorOutput(self, sess, epoch, gen_samples_at_epoch, n_samples_per_cond,
                             validX, trainX, trainCond, validCond,
                             x_raw, x_cond, x_random, x_synth, keep_prob, bn_train, loss_g_pre,
                             gen_output_train, gen_output_valid):


        sample_number = int(epoch / gen_samples_at_epoch)

        valid_diffs_to_round = []
        valid_diffs_to_exp = []
        valid_diffs_nonzero = []
        valid_round_sums = []
        valid_losses_g_pre = []
        valid_overlaps = []
        valid_correct_share = []
        valid_counter = Counter()

        train_diffs_to_round = []
        train_diffs_to_exp = []
        train_diffs_nonzero = []
        train_round_sums = []
        train_losses_g_pre = []
        train_overlaps = []
        train_correct_share = []
        train_counter = Counter()

        for i in range(n_samples_per_cond):
            n_synth = np.shape(validX)[0]
            randomX = np.random.normal(size=(n_synth, self.randomDim))
            x_synthetic_train,loss_g = sess.run([x_synth, loss_g_pre],
                                         feed_dict={x_raw: trainX[:n_synth, :], x_random: randomX,
                                                    x_cond: trainCond[:n_synth, :],
                                                    keep_prob: 1.0, bn_train: False})

            diff_to_round, diff_to_exp, diff_nonzero, round_sum, overlap, counter,correct_share = self.conditionalOutputAnalysis(x_synthetic_train,
                                                                                                 trainX[:n_synth,:])
            train_counter.update(counter)
            train_correct_share.extend(correct_share)
            train_diffs_to_round.extend(diff_to_round)
            train_diffs_to_exp.extend(diff_to_exp)
            train_diffs_nonzero.extend(diff_nonzero)
            train_round_sums.extend(round_sum)
            train_losses_g_pre.append(loss_g)
            train_overlaps.extend(overlap)


            x_synthetic_train = np.array(np.round(x_synthetic_train), dtype=np.int8)

            if epoch % gen_samples_at_epoch == 0:
                for j in range(n_synth):
                    gen_output_train[sample_number, j, i, :] = x_synthetic_train[j, :]

            randomX = np.random.normal(size=(n_synth, self.randomDim))
            x_synthetic_valid, loss_g = sess.run([x_synth, loss_g_pre],
                                         feed_dict={x_raw: validX, x_random: randomX, x_cond: validCond,
                                                    keep_prob: 1.0, bn_train: False})

            diff_to_round, diff_to_exp, diff_nonzero, round_sum, overlap, counter,correct_share = self.conditionalOutputAnalysis(x_synthetic_valid,
                                                                                                 validX)
            valid_counter.update(counter)
            valid_correct_share.extend(correct_share)
            valid_diffs_to_round.extend(diff_to_round)
            valid_diffs_to_exp.extend(diff_to_exp)
            valid_diffs_nonzero.extend(diff_nonzero)
            valid_round_sums.extend(round_sum)
            valid_losses_g_pre.append(loss_g)
            valid_overlaps.extend(overlap)


            x_synthetic_valid = np.array(np.round(x_synthetic_valid), dtype=np.int8)
            if epoch % gen_samples_at_epoch == 0:
                for j in range(n_synth):
                    gen_output_valid[sample_number, j, i, :] = x_synthetic_valid[j, :]

        gen_result_vec = [np.mean(train_diffs_to_round),
                          np.mean(valid_diffs_to_round),
                          np.mean(train_diffs_to_exp),
                          np.mean(valid_diffs_to_exp),
                          np.mean(train_diffs_nonzero),
                          np.mean(valid_diffs_nonzero),
                          np.mean(train_round_sums),
                          np.mean(valid_round_sums),
                          np.mean(train_losses_g_pre),
                          np.mean(valid_losses_g_pre),
                          np.mean(train_overlaps),
                          np.mean(valid_overlaps),
                          np.mean(train_correct_share),
                          np.mean(valid_correct_share)
                          ]

        return gen_result_vec, gen_output_train, gen_output_valid, train_counter, valid_counter

    def evalGAN(self,sess, trainX, validX, validCond, trainCond, x_raw, x_cond, x_random, keep_prob, bn_train, y_hats, x_synth):

        randomX_train = np.random.normal(size=(np.shape(trainX)[0], self.randomDim))
        randomX_valid = np.random.normal(size=(np.shape(validX)[0], self.randomDim))

        y_hat_real_single, y_hat_real_batch, y_hat_fake_single, y_hat_fake_batch, y_hat_random_single = y_hats

        feed_dict = {x_raw: trainX, x_cond: trainCond, x_random: randomX_train,keep_prob: 1.0, bn_train: False}

        y_hat_real_train, y_hat_fake_train, x_gen_train = sess.run([y_hat_real_single, y_hat_fake_single, x_synth],
                                                                   feed_dict=feed_dict)

        feed_dict = {x_raw: validX, x_cond: validCond, x_random: randomX_valid, keep_prob: 1.0, bn_train: False}

        y_hat_real_valid, y_hat_fake_valid, x_gen_valid = sess.run([y_hat_real_single, y_hat_fake_single, x_synth],
                                                                   feed_dict=feed_dict)

        y_hat_eval = [y_hat_real_train, y_hat_fake_train, y_hat_real_valid, y_hat_fake_valid]
        x_gen_eval = [x_gen_train, x_gen_valid]

        return y_hat_eval, x_gen_eval



    def mainTraining(self, sess, nEpochs, batchSize, trainX, validX, trainCond, validCond, keepProb,
                     x_raw, x_random, x_cond, x_synth, y_hats, keep_prob, bn_train,
                     optimize_d_after, optimize_g_after, loss_d_comb, loss_g_comb,
                     n_gen_samples, n_samples_per_cond,loss_g_pre,
                     discriminatorTrainPeriod, generatorTrainPeriod, logFile):

        #containers for generator output vectors
        gen_output_train = np.zeros([n_gen_samples, np.shape(validX)[0], n_samples_per_cond, np.shape(validX)[1]],
                                    dtype=np.int8)
        gen_output_valid = np.zeros([n_gen_samples, np.shape(validX)[0], n_samples_per_cond, np.shape(validX)[1]],
                                    dtype=np.int8)
        gen_samples_at_epoch = int(np.round(nEpochs / n_gen_samples))

        main_counter_valid = Counter()
        main_counter_train = Counter()

        if gen_samples_at_epoch==0: gen_samples_at_epoch=1

        #initialize lists containing the training results
        generator_training_status = []
        main_training_status = []

        for epoch in range(nEpochs):
            t1 = time.time()

            '''
            training step
            '''
            d_loss_vec, g_loss_vec = self.trainStep(sess,batchSize, discriminatorTrainPeriod, generatorTrainPeriod,
                                                    trainX, trainCond, optimize_d_after, optimize_g_after,
                                                    loss_d_comb, loss_g_comb, x_raw, x_random, x_cond,
                                                    keep_prob, keepProb, bn_train)

            '''
            validate training step
            '''

            validation_results = self.validStep(sess, trainX, validX, trainCond, validCond, batchSize, x_raw, x_cond,
                                                x_random, y_hats, keep_prob, bn_train)

            '''
            analyze generator output
            '''

            gen_results = self.validGeneratorOutput(sess, epoch, gen_samples_at_epoch, n_samples_per_cond,
                                                    validX, trainX, trainCond, validCond, x_raw, x_cond,x_random, x_synth,
                                                    keep_prob, bn_train,loss_g_pre, gen_output_train,gen_output_valid)

            #unpack gen_results as stat results and output vectors (train & valid)
            generator_result_vec, gen_output_train, gen_output_valid, counter_train, counter_valid = gen_results

            if (nEpochs-epoch)<100:
                main_counter_valid.update(counter_valid)
                main_counter_train.update(counter_train)

            '''
            update results lists which will later be saved
            '''
            delta_t = time.time() - t1
            generator_training_status.append(generator_result_vec)

            status = [epoch, delta_t, np.mean(d_loss_vec), np.mean(g_loss_vec)]
            status.extend(validation_results)
            main_training_status.append(status)


            '''
            print intermediary results in console and to file
            '''
            buf = 'epoch:%05d, time:%.2f d_loss:%.2f, g_loss:%.2f, acc_train:%.2f, acc_valid:%.2f, g_loss_pre_train:%.2f, g_loss_pre_valid:%.2f, overlap_train:%.2f, overlap_valid::%.2f' \
                  % (epoch, delta_t, np.mean(d_loss_vec), np.mean(g_loss_vec), validation_results[6],validation_results[2],
                     generator_result_vec[8], generator_result_vec[9], generator_result_vec[10], generator_result_vec[11])

            print(buf)
            self.print2file(buf, logFile)
        return main_training_status, generator_training_status, gen_output_train, gen_output_valid, main_counter_train, main_counter_valid


    def train(self,
              dataPath='data',
              modelPath='',
              outPath='out',
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0,
              keepProb=1.0,
              n_gen_samples=100,
              n_samples_per_cond=10,
              saveGeneratorSamples = False):

        '''
        initialize tensorflow variables
        '''
        #desired output for generator and part of the input for discriminator
        x_raw = tf.placeholder('float', [None, self.inputDim],name='raw')
        #condition for generator and other part of input for discriminator
        x_cond = tf.placeholder('float',[None, self.condDim],name='cond')
        #random input for generator
        x_random= tf.placeholder('float', [None, self.randomDim],name='rand')
        #random vector to train discriminator
        x_pre_rand = tf.placeholder('float', [None, self.inputDim], name='pre_rand')
        #keep probability for droupout layers
        keep_prob = tf.placeholder('float')
        #determines if generator uses batch normalisation
        bn_train = tf.placeholder('bool')
        #determines if discriminator gets batch average or only single sample
        single_mode = tf.placeholder('bool')

        '''
        build tensorflow graph for GAN
        '''
        #build autoencoder graph
        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        #build generator graph, x_fake is output of generator before decoding
        x_fake = self.buildGenerator(x_random, x_cond, bn_train)
        #build discriminator graph
        losses_pre, losses, y_hats, x_synth = self.buildDiscriminator(x_raw, x_fake, x_cond, x_pre_rand, keep_prob,
                                                                      decodeVariables, bn_train, single_mode)

        #unpack losses and y_hat (discriminator outputs)
        loss_g_pre, loss_d_pre = losses_pre
        loss_d, loss_g, loss_d_comb, loss_g_comb = losses

        # unpack tf variables used for training
        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # collect regularization losses
        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


        '''
        intitialize optimizers for each step
        '''
        # optimizer for gan pretraining
        optimize_d_pre = tf.train.AdamOptimizer().minimize(loss_d_pre + sum(all_regs), var_list=d_vars)
        optimize_g_pre = tf.train.AdamOptimizer().minimize(loss_g_pre + sum(all_regs), var_list=g_vars)

        # optimizer for autoencoder, discriminator and generator
        optimize_ae = tf.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d_after = tf.train.AdamOptimizer().minimize(loss_d_comb + sum(all_regs), var_list=d_vars)
        optimize_g_after = tf.train.AdamOptimizer().minimize(loss_g_comb + sum(all_regs),
                                                             var_list=g_vars + list(decodeVariables.values()))


        '''
        load training and validation data
        '''
        data = self.importTrainAndValidData(modelPath, dataPath, outPath)
        trainX, validX, trainCond, validCond, train_ind, valid_ind = data

        np.save(os.path.dirname(outPath) + '/train_ind.npy', train_ind)
        np.save(os.path.dirname(outPath) + '/valid_ind.npy', valid_ind)

        '''
        initialize tensorflow graph
        '''
        #initialize variables and saver
        initOp = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=saveMaxKeep)


        '''
        set up paths for saving
        '''
        #filepaths for output files
        saveFile = outPath + '_training_stats.npz'
        saveFileCounter = outPath + '_wrong_digits_counter.npz'
        saveFileLastEval = outPath +'_last_eval.npz'
        logFile = outPath+'.txt'


        #initialize lists which are not used without pretraining
        ae_training_status = []
        pre_train_losses = []

        with tf.Session() as sess:
            if modelPath == '': sess.run(initOp)
            else: saver.restore(sess, modelPath)

            print('')
            print('=================')
            print('train autoencoder')
            print('=================')
            print('')
            '''
            train autoencoder for pretrainEpochs epochs
            '''
            if modelPath== '':
                ae_training_status = self.pretrainAutoencoder(sess, pretrainEpochs, pretrainBatchSize, optimize_ae,
                                                              loss_ae, x_raw)

            print('')
            print('=================')
            print('pre training gan')
            print('=================')
            print('')

            '''
            pretrain generator with condition - expected results pairs and discriminator against random numbers
            '''
            if modelPath == '':
                 pretrain_results = self.pretrainGAN(sess, pretrainEpochs, batchSize, trainX, trainCond,validX, validCond,
                                                    optimize_d_pre, loss_d_pre, optimize_g_pre, loss_g_pre, x_raw, x_random,
                                                    x_pre_rand, x_cond, x_synth, keep_prob, keepProb, bn_train, y_hats,logFile)

                 pre_training_status, pre_generator_status,pre_counter_train, pre_counter_valid = pretrain_results

            print('')
            print('=================')
            print('main training gan')
            print('=================')
            print('')

            '''
            train gan and autoencoder for nEpochs epochs, with batches of size batchSize
            '''

            train_results = self.mainTraining(sess, nEpochs, batchSize, trainX, validX, trainCond, validCond, keepProb,
                                              x_raw, x_random, x_cond, x_synth, y_hats, keep_prob, bn_train,
                                              optimize_d_after, optimize_g_after, loss_d_comb, loss_g_comb,
                                              n_gen_samples, n_samples_per_cond,loss_g_pre,
                                              discriminatorTrainPeriod, generatorTrainPeriod, logFile)
            #unpack train results
            main_training_status, generator_status, gen_output_train, gen_output_valid, main_counter_train, main_counter_valid = train_results


            '''
            final evaluation of gan with all outputs from D and G
            '''
            y_hat_eval, x_gen_eval = self.evalGAN( sess, trainX, validX, validCond, trainCond, x_raw, x_cond, x_random, keep_prob,
                                                   bn_train, y_hats, x_synth)

            y_hat_real_train, y_hat_fake_train, y_hat_real_valid, y_hat_fake_valid = y_hat_eval
            x_gen_train, x_gen_valid =  x_gen_eval



            np.savez(saveFile,
                     ae_training_status=ae_training_status,
                     main_training_status=main_training_status,
                     generator_status=generator_status,
                     pre_training_status=pre_training_status,
                     pre_generator_status=pre_generator_status)

            np.savez(saveFileCounter,
                     pre_counter_train=pre_counter_train,
                     pre_counter_valid=pre_counter_valid,
                     main_counter_train=main_counter_train,
                     main_counter_valid=main_counter_valid
                     )

            np.savez(saveFileLastEval,
                     y_hat_real_train=y_hat_real_train,
                     y_hat_fake_train=y_hat_fake_train,
                     y_hat_real_valid=y_hat_real_valid,
                     y_hat_fake_valid=y_hat_fake_valid,
                     x_gen_train=x_gen_train,
                     x_gen_valid=x_gen_valid
                     )

            if saveGeneratorSamples:
                np.save(outPath+'_gen_out_train.npy', gen_output_train)
                np.save(outPath + '_gen_out_valid.npy', gen_output_valid)
            savePath = saver.save(sess, outPath)

        print('data saved at:', savePath)


def parse_arguments(parser):
    parser.add_argument('--embed_size', type=int, default=128, help='The dimension size of the embedding, which will be generated by the generator. (default value: 128)')
    parser.add_argument('--noise_size', type=int, default=128, help='The dimension size of the random noise, on which the generator is conditioned. (default value: 128)')
    parser.add_argument('--generator_size', type=tuple, default=(256, 256), help='The dimension size of the generator. Note that another layer of size "--embed_size" is always added. (default value: (128, 128))')
    parser.add_argument('--discriminator_size', type=tuple, default=(512, 256, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    parser.add_argument('--compressor_size', type=tuple, default=(256), help='The dimension size of the encoder of the autoencoder. Note that another layer of size "--embed_size" is always added. Therefore this can be a blank tuple. (default value: ())')
    parser.add_argument('--decompressor_size', type=tuple, default=(512), help='The dimension size of the decoder of the autoencoder. Note that another layer, whose size is equal to the dimension of the <patient_matrix>, is always added. Therefore this can be a blank tuple. (default value: ())')
    parser.add_argument('--data_type', type=str, default='binary', choices=['binary', 'count'], help='The input data type. The <patient matrix> could either contain binary values or count values. (default value: "binary")')
    parser.add_argument('--batchnorm_decay', type=float, default=0.99, help='Decay value for the moving average used in Batch Normalization. (default value: 0.99)')
    parser.add_argument('--L2', type=float, default=0.001, help='L2 regularization coefficient for all weights. (default value: 0.001)')

    parser.add_argument('--data_file', type=str, default='cosmic_patient_dataset_n_hot_not_single.npy', metavar='<patient_matrix>', help='The path to the numpy matrix containing aggregated patient records.')
    parser.add_argument('--out_file', type=str, default='output', metavar='<out_file>', help='The path to the output models.')
    parser.add_argument('--model_file', type=str, metavar='<model_file>', default='', help='The path to the model file, in case you want to continue training. (default value: '')')
    parser.add_argument('--n_pretrain_epoch', type=int, default=100, help='The number of epochs to pre-train the autoencoder. (default value: 100)')
    parser.add_argument('--n_epoch', type=int, default=1000, help='The number of epochs to train medGAN. (default value: 1000)')
    parser.add_argument('--n_discriminator_update', type=int, default=2, help='The number of times to update the discriminator per epoch. (default value: 2)')
    parser.add_argument('--n_generator_update', type=int, default=1, help='The number of times to update the generator per epoch. (default value: 1)')
    parser.add_argument('--pretrain_batch_size', type=int, default=100, help='The size of a single mini-batch for pre-training the autoencoder. (default value: 100)')
    parser.add_argument('--batch_size', type=int, default=1000, help='The size of a single mini-batch for training medGAN. (default value: 1000)')
    parser.add_argument('--save_max_keep', type=int, default=0, help='The number of models to keep. Setting this to 0 will save models for every epoch. (default value: 0)')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='Keep probability for Dropout in Discriminator')
    parser.add_argument('--n_gen_samples', type=int, default=100, help='Number of generator samples taken during training (default value: 100)')
    parser.add_argument('--n_samples_per_cond', type=int, default=10, help='Number of generator outputs per condition (default value: 10)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data = np.load(args.data_file+'_treat.npy')
    data_cond = np.load(args.data_file+'_demo.npy')
    inputDim = data.shape[1]
    condDim = data_cond.shape[1]

    embed_size = 64
    noise_size = 210
    generator_size = [256, 128, 64]
    discriminator_size = [512, 256, 128, 1]
    compressor_size = []
    decompressor_size = []


    tg = treatGAN(dataType=args.data_type,
                inputDim=inputDim,
                condDim=condDim,
                embeddingDim=embed_size,
                randomDim=noise_size,
                generatorDims=generator_size,
                discriminatorDims=discriminator_size,
                compressDims=compressor_size,
                decompressDims=decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)

    tg.train(dataPath=args.data_file,
             modelPath=args.model_file,
             outPath=args.out_file,
             pretrainEpochs=args.n_pretrain_epoch,
             nEpochs=args.n_epoch,
             discriminatorTrainPeriod=args.n_discriminator_update,
             generatorTrainPeriod=args.n_generator_update,
             pretrainBatchSize=args.pretrain_batch_size,
             batchSize=args.batch_size,
             saveMaxKeep=args.save_max_keep,
             keepProb=args.keep_prob,
             n_gen_samples=args.n_gen_samples,
             n_samples_per_cond=args.n_samples_per_cond)
