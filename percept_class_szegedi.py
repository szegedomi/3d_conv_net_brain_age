# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:45:12 2018

learning from skull size

@author: latlab
"""

#%%
#importing necessary libraries

import numpy as np
import tensorflow as tf
import random as rn
from sklearn.cross_validation import KFold
from sklearn import preprocessing

#%%
#defining accuracy as a measure of leraning quality

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

  
  
#%%
#Defining tensorflow Graph for learning purpose

#Constants


num_class = 2   #number of classes
labels=np.asarray(range(num_class)) #labels for classification
num_labels = 2  #number of classes
num_features = 3

num_samples=60  #number of samples
num_kmeans=6   #number of folds in cross validation


num_hid_1 = 32  #number of neurons in the first fully connected hidden layer
num_hid_2 = 16
beta = 0.01 #parameter for the regulizer


graph=tf.Graph()

with graph.as_default():
    
    #Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=((num_samples-num_samples/num_kmeans), num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=((num_samples-num_samples/num_kmeans), num_labels))
    tf_valid_dataset = tf.placeholder(tf.float32, name="valid", shape=((num_samples // num_kmeans), num_features))
    tf_keep_prob =  tf.placeholder(tf.float32, name="keep_prob")

    
    
    #Variables
    layer1_weights = tf.Variable(tf.truncated_normal(
        [3,num_hid_1], stddev=0.1))
    layer1_biases = tf.Variable(tf.constant(1.0, shape=[num_hid_1]))
    
    layer2_weights = tf.Variable(tf.truncated_normal(
        [num_hid_1,num_hid_2], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hid_2]))
    
    layer3_weights = tf.Variable(tf.truncated_normal(
        [num_hid_2, num_labels], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    #model
    def model(data):
        fully_connect = tf.nn.relu(tf.matmul(data, layer1_weights) + layer1_biases)
        fully_connect_drop = tf.nn.dropout(fully_connect, tf_keep_prob)
        fully_connect = tf.nn.relu(tf.matmul(fully_connect_drop, layer2_weights) + layer2_biases)
        fully_connect_drop = tf.nn.dropout(fully_connect, tf_keep_prob)
        
        return tf.matmul(fully_connect_drop, layer3_weights) + layer3_biases
        
    #computation of loss
    
    logits = model(tf_train_dataset)
    #regulizers=tf.nn.l2_loss(layer3_weights)+tf.nn.l2_loss(layer4_weights)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

    #optimizer
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    #predictions
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    
    
#%%

import time
start = time.time()

#running the training and computing the loss on the test data afterwards
num_steps=1501
keep_prob = 1 #keeping probability of a neuron when dropout applied, default value is 1
num_kmeans=6


#Creating file for execution time log
file = open('D:\\Domonkos\\resting_raw\\basic_network\\' + str(num_kmeans) + "_" + str(num_steps) + "_" + str(keep_prob) + "_" + "_skull_adam_10times_6means_rs_plane.txt",  "w")
file.write("execution_time" + " " + "minibatch_loss" + " " + "minibatch_accuracy" + " " + "validation_accuracy" + " " + "num_kmeans" + " " + "num_step" + " " + '\n')


#labels coding: Col1: Patient number Col2: Patient ID Col3: Age class (0: young 1: old) Col4: Gender (0:Male, 1: Female) Col5: Activity (1:sport, 2:choir, 3:other) Col6: Age
metadata=np.loadtxt("D:\\Domonkos\\resting_raw\\metadata\\labels3.txt")


for k in range(10):
    #Splitting data of the patient IDs to train and validation sets for cross validation
    kfold=KFold(num_samples, num_kmeans, True, None)
    ind1 = 0
    print('Initialized' + str(k) + '\n')
    for train_index, test_index  in kfold:
        np.random.shuffle(train_index)
        
        #building up the validation dataset ATTENTION iteration is going backwards from the end of the test_index list for the data and for the laabels as weel so they are consistent
        valid_dataset=metadata[test_index,6:9]
        valid_labels=metadata[test_index, 2]
        valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
    
        train_dataset=metadata[train_index,6:9]
        train_labels=metadata[train_index, 2]
        train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
        
        scaler = preprocessing.StandardScaler().fit(train_dataset)
        train_dataset=scaler.transform(train_dataset)
        valid_dataset=scaler.transform(valid_dataset)
        
        with tf.Session(graph=graph) as session:
            #initializing graph
            tf.global_variables_initializer().run()
            print('Initialized')
            
            #iterating over optimization steps
            for step in range(num_steps):
                #creating the variables of the graph and running the training
                feed_dict = {tf_train_dataset : train_dataset, tf_train_labels : train_labels, tf_valid_dataset : valid_dataset, tf_keep_prob : keep_prob}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                file.write(str(time.time()-start) + " " + str(l) + " " + str(accuracy(predictions, train_labels)) + " " + str(accuracy(valid_prediction.eval(feed_dict = feed_dict ), valid_labels)) + " " + str(ind1) + " " + str(step) + " " + '\n')
                
            print(accuracy(valid_prediction.eval(feed_dict = feed_dict ), valid_labels))
        ind1+=1


end = time.time()
file.write(str(end-start))
file.close() 
