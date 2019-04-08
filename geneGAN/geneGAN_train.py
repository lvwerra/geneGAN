# noinspection PyInterpreter
import sys, time, argparse
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.contrib.layers import batch_norm

_VALIDATION_RATIO = 0.1


class geneGAN(object):

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
        '''
        print()
        print('configuration:')
        print('---------------------------------')
        print('input dim:\t', self.inputDim)
        print('random dim:\t', self.randomDim)
        print('embedding:\t', self.embeddingDim)
        print('compressor:\t', self.compressDims)
        print('decompressor:\t', self.decompressDims)
        print('generator:\t', self.generatorDims)
        print('discriminator:\t', self.discriminatorDims)
        print('---------------------------------')
        print()
        '''

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

    def buildGenerator(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator'):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim], regularizer=l2_regularizer(self.l2scale))
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
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
    
    def buildGeneratorTest(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator'):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim], regularizer=l2_regularizer(self.l2scale))
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
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
    
    def getDiscriminatorResults(self, x_input, keepRate, single_mode, reuse=False):

        batchSize = tf.shape(x_input)[0]
        inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_input, 0), [batchSize]), (batchSize, self.inputDim))

        tempVec = tf.cond(single_mode,
                          lambda: tf.concat([x_input, tf.zeros([batchSize, self.inputDim]), tf.ones([batchSize, 1])], 1),
                          lambda: tf.concat([x_input, inputMean, tf.zeros([batchSize,1])], 1))

        tempDim = self.inputDim * 2 + 1
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
    
    def buildDiscriminator(self, x_real, x_fake, keepRate, decodeVariables):
        '''
        buildDiscriminator builds the Discriminator Tensorflow graph and returns variables used for training
        :param x_real: real data sample
        :param x_fake: generator output (before AE)
        :param keepRate: keeprate for dropout
        :param decodeVariables: variables to decode generator output
        :return: losses dictionary, discriminator predictions for real and fake data in single and batch mode, decoded generator output
        '''
        const_true = tf.constant(True, dtype=tf.bool)
        const_false = tf.constant(False, dtype=tf.bool)

        #Discriminate for real samples
        y_hat_real_single = self.getDiscriminatorResults(x_real, keepRate, const_true, reuse=False)
        y_hat_real_batch = self.getDiscriminatorResults(x_real, keepRate, const_false, reuse=True)

        #Decompress, then discriminate for fake samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        y_hat_fake_single = self.getDiscriminatorResults(x_decoded, keepRate, const_true, reuse=True)
        y_hat_fake_batch = self.getDiscriminatorResults(x_decoded, keepRate, const_false, reuse=True)

        #define D and G losses in single and batch mode
        loss_d_single = -tf.reduce_mean(tf.log(y_hat_real_single + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake_single + 1e-12))
        loss_d_batch  = -tf.reduce_mean(tf.log(y_hat_real_batch + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake_batch + 1e-12))
        loss_g_single = -tf.reduce_mean(tf.log(y_hat_fake_single + 1e-12))
        loss_g_batch = -tf.reduce_mean(tf.log(y_hat_fake_batch + 1e-12))

        loss_d = loss_d_batch
        loss_g = loss_g_batch

        loss_d_comb = loss_d_batch + loss_d_single
        loss_g_comb = loss_g_batch + loss_g_single

        losses = {}
        losses['d_batch'] = loss_d
        losses['g_batch'] = loss_g
        losses['d_comb'] = loss_d_comb
        losses['g_comb'] = loss_g_comb

        y_hats  = {}
        y_hats['real_single'] = y_hat_real_single
        y_hats['real_batch'] = y_hat_real_batch
        y_hats['fake_single'] = y_hat_fake_single
        y_hats['fake_batch'] = y_hat_fake_batch

        return losses, y_hats, x_decoded

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

    def train(self,
              dataPath='data',
              autoencoderData='',
              modelPath='',
              outPath='out',
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0,
              keepProb=1.0):

        #discriminator_input
        x_raw = tf.placeholder('float', [None, self.inputDim])
        #generator_input
        x_random= tf.placeholder('float', [None, self.randomDim])
        #keep probability for dropout
        keep_prob = tf.placeholder('float')
        #batchnormalization
        bn_train = tf.placeholder('bool')

        #dataset with all possible combinations for autoencoder pre-training

        #initialize AE, G and D
        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        x_fake = self.buildGenerator(x_random, bn_train)
        losses, y_hats, x_synth = self.buildDiscriminator(x_raw, x_fake, keep_prob, decodeVariables)

        if modelPath=='':
            #new model
            trainX, validX, train_ind, valid_ind = self.loadData(dataPath)

        else:
            #continue training of existing model, load validation and training indices
            data = np.load(dataPath)
            train_ind = np.load(os.path.dirname(modelPath)+'/train_ind.npy')
            valid_ind = np.load(os.path.dirname(modelPath) + '/valid_ind.npy')
            trainX, validX = data[train_ind, :], data[valid_ind, :]

        #save indices of valid/train part of dataset
        np.save(os.path.dirname(outPath) + '/train_ind.npy', train_ind)
        np.save(os.path.dirname(outPath) + '/valid_ind.npy', valid_ind)
        if len(autoencoderData)<1:
            autoencoderData = dataPath
        
        ae_trainX, ae_validX, _, _ = self.loadData(autoencoderData)

        #load pairs not included in the dataset:
        pairsOutData = np.load(dataPath[:-7] + '_out.npy')

        #initialize the optimizers for each GAN part
        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        optimize_ae = tf.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(losses['d_comb'] + sum(all_regs), var_list=d_vars)
        optimize_g = tf.train.AdamOptimizer().minimize(losses['g_comb'] + sum(all_regs),
                                                             var_list=g_vars + list(decodeVariables.values()))

        initOp = tf.global_variables_initializer()
        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))

        saver = tf.train.Saver(max_to_keep=saveMaxKeep)
        saveFile = outPath + '_training_stats.npz'
        logFile = outPath+'.txt'

        generator_training_status = []
        generator_tuples = []
        main_training_status = []
        ae_training_status = []
        quality_status = []
        all_scores_scaled = []
        all_scores_unscaled = []
        best_epoch_scaled = 0
        best_epoch_unscaled = 0

        with tf.Session() as sess:
            if modelPath == '': sess.run(initOp)
            else: saver.restore(sess, modelPath)

            '''
            train autoencoder for pretrainEpochs epochs
            '''

            nTrainBatches = int(np.ceil(float(ae_trainX.shape[0])) / float(pretrainBatchSize))
            nValidBatches = int(np.ceil(float(ae_validX.shape[0])) / float(pretrainBatchSize))

            if modelPath== '':
                for epoch in range(pretrainEpochs):
                    t1 = time.time()
                    idx = np.random.permutation(ae_trainX.shape[0])
                    trainLossVec = []
                    for i in range(nTrainBatches):
                        batchX = ae_trainX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw:batchX})
                        trainLossVec.append(loss)
                    idx = np.random.permutation(ae_validX.shape[0])
                    validLossVec = []

                    for i in range(nValidBatches):
                        batchX = ae_validX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        loss = sess.run(loss_ae, feed_dict={x_raw:batchX})
                        validLossVec.append(loss)
                    validReverseLoss = 0.
                    buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f, validReverseLoss:%f' % (epoch, np.mean(trainLossVec), np.mean(validLossVec), validReverseLoss)
                    print(buf)
                    t2 = time.time()
                    delta_t = t2-t1
                    t1 = t2
                    ae_training_status.append([epoch, delta_t, np.mean(trainLossVec), np.mean(validLossVec)])

                    #self.print2file(buf, logFile)


            '''
            train gan and autoencoder for nEpochs epochs, with bachtches of size batchSize
            '''

            idx = np.arange(trainX.shape[0])

            for epoch in range(nEpochs):
                t1 = time.time()
                d_loss_vec = []
                g_loss_vec = []

                for i in range(nBatches):
                    for _ in range(discriminatorTrainPeriod):
                        tBatchSize = min(batchSize, len(idx))
                        batchIdx = np.random.choice(idx, size=tBatchSize, replace=False)
                        batchX = trainX[batchIdx]
                        randomX = np.random.normal(size=(tBatchSize, self.randomDim))
                        _, discLoss = sess.run([optimize_d, losses['d_comb']],
                                               feed_dict={x_raw: batchX, x_random: randomX, keep_prob: keepProb,
                                                          bn_train: False})
                        d_loss_vec.append(discLoss)
                    for _ in range(generatorTrainPeriod):
                        randomX = np.random.normal(size=(tBatchSize, self.randomDim))
                        _, generatorLoss = sess.run([optimize_g, losses['g_comb']],
                                                    feed_dict={x_raw: batchX, x_random: randomX, keep_prob: 1.0,
                                                               bn_train: True})
                        g_loss_vec.append(generatorLoss)


                '''
                validate training step
                '''

                idx_valid = np.arange(len(validX))
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

                    randomX = np.random.normal(size=(vBatchSize, self.randomDim))
                    preds_real_single, preds_real_batch, preds_fake_single, preds_fake_batch, = sess.run(
                        [y_hats['real_single'], y_hats['real_batch'],
                         y_hats['fake_single'], y_hats['fake_batch']],
                        feed_dict={x_raw: batchX_valid,
                                   x_random: randomX,
                                   keep_prob: 1.0,
                                   bn_train: False})

                    validAcc = self.calculateDiscAccuracy(preds_real_batch, preds_fake_batch)
                    validAuc = self.calculateDiscAuc(preds_real_batch, preds_fake_batch)
                    validAccVec.append(validAcc)
                    validAucVec.append(validAuc)

                    validAcc = self.calculateDiscAccuracy(preds_real_single, preds_fake_single)
                    validAuc = self.calculateDiscAuc(preds_real_single, preds_fake_single)
                    validAccVec_single.append(validAcc)
                    validAucVec_single.append(validAuc)

                    ### validate with training data ###
                   
                    vBatchSize = tBatchSize
                    batchIdx_train = np.random.choice(idx, size=vBatchSize, replace=False)
                    batchX_train = trainX[batchIdx_train]

                    randomX = np.random.normal(size=(vBatchSize, self.randomDim))
                    preds_real_single, preds_real_batch, preds_fake_single, preds_fake_batch, = sess.run(
                        [y_hats['real_single'], y_hats['real_batch'],
                         y_hats['fake_single'], y_hats['fake_batch']],
                        feed_dict={x_raw: batchX_train,
                                   x_random: randomX,
                                   keep_prob: 1.0,
                                   bn_train: False})

                    validAcc = self.calculateDiscAccuracy(preds_real_batch, preds_fake_batch)
                    validAuc = self.calculateDiscAuc(preds_real_batch, preds_fake_batch)
                    trainAccVec.append(validAcc)
                    trainAucVec.append(validAuc)

                    validAcc = self.calculateDiscAccuracy(preds_real_single, preds_fake_single)
                    validAuc = self.calculateDiscAuc(preds_real_single, preds_fake_single)
                    trainAccVec_single.append(validAcc)
                    trainAucVec_single.append(validAuc)




                preds_in_train_s, preds_in_train_b = sess.run([y_hats['real_single'], y_hats['real_batch']], feed_dict={x_raw: trainX,
                                                                                                    keep_prob: 1.0,
                                                                                                    bn_train: False})

                preds_in_valid_s, preds_in_valid_b = sess.run([y_hats['real_single'], y_hats['real_batch']], feed_dict={x_raw: validX,
                                                                                                    keep_prob: 1.0,
                                                                                                    bn_train: False})

                preds_out_s, preds_out_b = sess.run([y_hats['real_single'], y_hats['real_batch']],
                                                    feed_dict={x_raw: pairsOutData,
                                                               keep_prob: 1.0,
                                                               bn_train: False})

                u_s_train, u_s_fake, u_score_train, b_s_train, b_s_fake, b_score_train, b_bound = self.calculateDiscQuality(preds_in_train_s, preds_out_s)
                u_s_valid, _ , u_score_valid, b_s_valid, _, b_score_valid, _ = self.calculateDiscQuality(preds_in_valid_s, preds_out_s, boundary=b_bound)


                synth_samples = 10000
                randomX = np.random.normal(size=(synth_samples, self.randomDim))
                x_synthetic = sess.run(x_synth, feed_dict={x_random: randomX, keep_prob: 1.0, bn_train: False})

                gen_nbr_valid, gen_nbr_valid_unique, gen_valid_tuples = self.countValidPairs(x_synthetic)

                t2 = time.time()
                delta_t = t2 - t1
                t1 = t2
                buf = 'Epoch:%05d, time:%.2f d_loss:%.2f, g_loss:%.2f, acc:%.2f, score_t:%.2f, score_v:%.2f, gen_v:%d' % (
                    epoch, delta_t, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec),
                    b_score_train, b_score_valid, gen_nbr_valid_unique)
                print(buf)

                main_training_status.append([epoch, delta_t,
                                             np.mean(d_loss_vec),
                                             np.mean(g_loss_vec),
                                             np.mean(validAccVec),
                                             np.mean(validAucVec),
                                             np.mean(validAccVec_single),
                                             np.mean(validAucVec_single),
                                             np.mean(trainAccVec),
                                             np.mean(trainAucVec),
                                             np.mean(trainAccVec_single),
                                             np.mean(trainAucVec_single),
                                             np.mean(np.concatenate((preds_in_train_s, preds_in_valid_s))),
                                             np.mean(preds_out_s),
                                             np.mean(np.concatenate((preds_in_train_b, preds_in_valid_b))),
                                             np.mean(preds_out_b)
                                             ])
                quality_status.append([u_s_train, u_s_valid, u_s_fake, u_score_train, u_score_valid,
                                       b_s_train, b_s_valid, b_s_fake, b_score_train, b_score_valid,
                                       b_bound])

                generator_training_status.append([gen_nbr_valid, gen_nbr_valid_unique])
                generator_tuples.append(gen_valid_tuples)

                if epoch>0 and b_score_valid<min(all_scores_scaled):
                    best_epoch_scaled = epoch
                    savePath = saver.save(sess, outPath + '_best_scaled')

                if epoch>0 and u_score_valid<min(all_scores_unscaled):
                    best_epoch_unscaled = epoch
                    savePath = saver.save(sess, outPath+'_best_unscaled')

                all_scores_scaled.append(b_score_valid)
                all_scores_unscaled.append(u_score_valid)



                self.print2file(buf, logFile)

            np.savez(saveFile,
                     ae_training_status=ae_training_status,
                     main_training_status=main_training_status,
                     quality_status=quality_status,
                     generator_training_status=generator_training_status)
                     #generator_tuples=generator_tuples)

            savePath = saver.save(sess, outPath)#, global_step=epoch)

        print(savePath)
        print('best epoch scaled:', best_epoch_scaled)
        print('best epoch unscaled:', best_epoch_unscaled)


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


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data = np.load(args.data_file)
    inputDim = data.shape[1]

    embed_size = 256
    noise_size = 256
    generator_size = [256, 256]
    discriminator_size = [256, 128, 1]
    compressor_size = []
    decompressor_size = []

    mg = geneGAN(dataType=args.data_type,
                inputDim=inputDim,
                embeddingDim=embed_size,
                randomDim=noise_size,
                generatorDims=generator_size,
                discriminatorDims=discriminator_size,
                compressDims=compressor_size,
                decompressDims=decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)

    mg.train(dataPath=args.data_file,
             autoencoderData='',
             modelPath=args.model_file,
             outPath=args.out_file,
             pretrainEpochs=args.n_pretrain_epoch,
             nEpochs=args.n_epoch,
             discriminatorTrainPeriod=args.n_discriminator_update,
             generatorTrainPeriod=args.n_generator_update,
             pretrainBatchSize=args.pretrain_batch_size,
             batchSize=args.batch_size,
             saveMaxKeep=args.save_max_keep,
             keepProb=args.keep_prob)

    # To generate synthetic data using a trained model:
    # Comment the train function above and un-comment generateData function below.
    # You must specify "--model_file" and "<out_file>" to generate synthetic data.
    #mg.generateData(nSamples=10000,
                    #modelFile=args.model_file,,
                    #batchSize=args.batch_size,
                    #outFile=args.out_file)
