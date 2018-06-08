# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:18:02 2018

training for regression resampled in house data

@author: latlab
"""



#%%
#importing necessary libraries

import nibabel as nib
import os
import numpy as np
import tensorflow as tf
import random as rn
from sklearn.cross_validation import KFold



        
        
#%%
#defining accuracy as a measure of leraning quality

def R_squared(predictions, labels):
  return 1-(np.mean(np.square(predictions-labels))/np.mean(np.square(labels-np.mean(labels))))
  

  
  
#%%
#Defining tensorflow Graph for learning purpose

#Constants


img_size_x = 64    #image size in x direction
img_size_y = 64    #image size in y direction
img_size_z = 28  #image size in z direction (depth)
num_channels = 1    #number of channels at a certain pixel
num_labels = 1  #output dimension

batch_size = 4  #number of elemnets in a batch
num_samples=60  #number of samples
num_kmeans=10   #number of folds in cross validation

kernel_size_1 = 5   #size of the first sqared convolution kernel
depth_1 = 5 #depth of the first convolution kernel
num_hid_1 = 64  #number of neurons in the first fully connected hidden layer
beta = 0.01 #parameter for the regulizer



graph=tf.Graph()


with graph.as_default():
    
    #Input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, img_size_x, img_size_y, img_size_z, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.placeholder(tf.float32, name="valid", shape=((num_samples // num_kmeans), img_size_x, img_size_y, img_size_z, num_channels))
    tf_keep_prob =  tf.placeholder(tf.float32, name="keep_prob")
    
    #Variables
    layer1_weights =tf.get_variable("layer1_weights", shape = [kernel_size_1, kernel_size_1, kernel_size_1, num_channels, depth_1], initializer=tf.contrib.layers.xavier_initializer())
    layer1_biases = tf.Variable(tf.zeros([depth_1]))
    
    layer2_weights =tf.get_variable("layer2_weights", shape= [kernel_size_1, kernel_size_1, kernel_size_1, depth_1, depth_1], initializer=tf.contrib.layers.xavier_initializer())
    layer2_biases = tf.Variable(tf.constant(0.001, shape=[depth_1]))
    
    layer3_weights = tf.get_variable("layer3_weights", shape=[(img_size_x // 4) * (img_size_y // 4) * (img_size_z // 4 ) * depth_1, num_hid_1], initializer=tf.contrib.layers.xavier_initializer())
    layer3_biases = tf.Variable(tf.constant(0.001, shape=[num_hid_1]))
    
    layer4_weights = tf.get_variable("layer4_weights", shape=[num_hid_1, num_labels], initializer = tf.contrib.layers.xavier_initializer())
    layer4_biases = tf.Variable(tf.constant(0.001, shape=[num_labels]))
    
  
    
    #model
    def model(data, tf_keep_prob):
        conv = tf.nn.conv3d(data, layer1_weights, [1, 1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        pool_pre = tf.nn.max_pool3d(hidden, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')
        pool = tf.nn.dropout(pool_pre, tf_keep_prob)
        conv = tf.nn.conv3d(pool, layer2_weights, [1, 1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        pool_pre = tf.nn.max_pool3d(hidden, [1,2,2,2,1], [1,2,2,2,1], padding='SAME')
        pool = tf.nn.dropout(pool_pre, tf_keep_prob)
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3] * shape[4]])
        fully_connect_1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        fully_connect_1_drop = tf.nn.dropout(fully_connect_1, tf_keep_prob)

        return tf.matmul(fully_connect_1_drop, layer4_weights) + layer4_biases
        
        
    #computation of loss
    
    prediction = model(tf_train_dataset, tf_keep_prob)
    loss = tf.contrib.losses.mean_squared_error(labels=tf_train_labels, predictions= prediction)
    
    
    #optimizer
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss) #optional and simpler optimizer

    
    #predictions
    train_prediction = model(tf_train_dataset, tf_keep_prob)
    valid_prediction = model(tf_valid_dataset,1)

    
    
    
#%%
import time
start = time.time()

#running the training and computing the loss on the test data afterwards
num_steps=3001
keep_prob = 0.85 #keeping probability of a neuron when dropout applied, default value is 1
num_kmeans=10 #60 samples > 6 validation samples in a fold for cross validation

#Creating file for execution time log
file = open('D:\\Domonkos\\resting_raw\\regression\\' + str(num_kmeans) + "_" + str(num_steps) + "_" + str(keep_prob) + "_" + str(batch_size) + "_regr_resamp1.txt",  "w")
file.write("execution_time" + " " + "minibatch_loss" + " " + "minibatch_R_squared" + " " + "validation_R_squared" + " " + "num_kmeans" + " " + "num_step"  + " " + '\n')



#labels coding: Col1: Patient number Col2: Patient ID Col3: Age class (0: young 1: old) Col4: Gender (0:Male, 1: Female) Col5: Activity (1:sport, 2:choir, 3:other) Col6: Age
metadata=np.loadtxt("D:\\Domonkos\\resting_raw\\metadata\\labels.txt")



validation_labels=[]
validation_predictions=[]
validation_indices=[]



ind1 = 0

#Splitting data of the patient IDs to train and validation sets for cross validation
kfold=KFold(num_samples, num_kmeans, True, None)


for train_index, test_index  in kfold:
    np.random.shuffle(train_index)
   
    
    #building up the validation dataset ATTENTION iteration is going backwards from the end of the test_index list for the data and for the laabels as weel so they are consistent
    valid_dataset=[]
    valid_labels=[]
    for j in range(test_index.shape[0]):
        valid_dataset.append(np.load("D:\\Domonkos\\resting_raw\\numpys_resamp_ss\\rss_" + str(test_index[-(j+1)]) + "_" + str(int(np.floor(rn.random() * 100))) + ".npy"))
        valid_labels.append(metadata[test_index[-(j+1)],5]/90)
    valid_dataset = np.asarray(valid_dataset)
    valid_labels = np.reshape(np.asarray(valid_labels), (num_samples/num_kmeans,1))
    validation_labels.append(valid_labels)
    validation_indices.append(test_index)
    with tf.Session(graph=graph) as session:
        #initializing graph
        tf.global_variables_initializer().run()
        print('Initialized')
        
        #iterating over optimization steps
        for step in range(num_steps):
            #building up the batch dataset
            offset = (step * batch_size) % ((num_samples - num_samples/num_kmeans) - batch_size)
            batch_data=[]
            batch_labels=[]
            for i in range(batch_size):
                ind=train_index[(offset+i) % train_index.shape[0]]
                batch_data.append(np.load("D:\\Domonkos\\resting_raw\\numpys_resamp_ss\\rss_" + str(ind) + "_" + str(int(np.floor(rn.random() * 100))) + ".npy"))
                batch_labels.append(metadata[ind,5]/90)
            batch_data = np.asarray(batch_data)
            batch_labels = np.reshape(np.asarray(batch_labels), (batch_size,1))
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_valid_dataset : valid_dataset, tf_keep_prob : keep_prob}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            
            
            file.write(str(time.time()-start) + " " + str(l) + " " + str(R_squared(predictions,batch_labels)) + " " + str(R_squared(valid_prediction.eval(feed_dict = feed_dict ), valid_labels)) + " " + str(ind1) + " " + str(step) + " "  + '\n')
            if(step % 20 ==0):
                print("Minibatch loss at step " + str(step) + "= " + str(l))
        validation_predictions.append(valid_prediction.eval(feed_dict = feed_dict))
    
            
    ind1+=1
               
            
validation_predictions=np.asarray(validation_predictions)
validation_labels=np.asarray(validation_labels)
validation_indices=np.asarray(validation_indices)
    
end = time.time()
file.write(str(end-start))
file.close() 

#%%
