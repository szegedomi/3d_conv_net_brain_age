# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:59:10 2018

transfer learning conv const fully init

@author: latlab
"""


import numpy as np
import tensorflow as tf
import random as rn
from sklearn.cross_validation import KFold

#%%
#defining accuracy as a measure of leraning quality

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

#%%
#Defining tensorflow Graph for learning purpose on external database

#Constants


num_class = 2   #number of classes
labels=np.asarray(range(num_class)) #labels for classification
img_size_x = 64    #image size in x direction
img_size_y = 64    #image size in y direction
img_size_z = 28 #image size in z direction (depth)
num_channels = 1    #number of channels at a certain pixel
num_labels = 2  #number of classes  


batch_size1 = 4  #number of elemnets in a batch
num_samples1=92  #number of samples


kernel_size_1 = 5   #size of the first sqared convolution kernel
depth_1 = 5 #depth of the first convolution kernel
num_hid_1 = 64  #number of neurons in the first fully connected hidden layer
beta = 0.01 #parameter for the regulizer




    
    

#%%

import time
start = time.time()

#running the training and computing the loss on the test data afterwards
num_steps1=1501
keep_prob1 = 1 #keeping probability of a neuron when dropout applied, default value is 1


#Creating file for execution time log
file = open('D:\\Domonkos\\transfer_learning\\'  + str(num_steps1) + "_" + str(keep_prob1) + "_" + str(batch_size1) + "_transferConstInitprior_adam.txt",  "w")
file.write("execution_time" + " " + "minibatch_loss" + " " + "minibatch_accuracy" + " "+ " " + "num_step" + " " + '\n')



num_kmeans=10 
num_steps2=151
keep_prob2 = 1 #keeping probability of a neuron when dropout applied, default value is 1
batch_size2 = 4  #number of elemnets in a batch
#Creating file for execution time log
file2 = open('D:\\Domonkos\\transfer_learning\\' + str(num_kmeans) + "_" + str(num_steps2) + "_" + str(keep_prob2) + "_" + str(batch_size2) + "_transferConstInitposterior_adam.txt",  "w")
file2.write("execution_time" + " " + "minibatch_loss" + " " + "minibatch_accuracy" + " " + "validation_accuracy" + " " + "num_kmeans" + " " + "num_step" + " " + '\n')

#labels coding: Col1: Patient number Col2: Patient ID Col3: Age class (0: young 1: old) Col4: Gender (0:Male, 1: Female) Col5: Activity (1:sport, 2:choir, 3:other) Col6: Age
metadata=np.loadtxt("D:\\Domonkos\\transfer_learning\\labels_tf.txt").astype(int)
#labels coding: Col1: Patient number Col2: Patient ID Col3: Age class (0: young 1: old) Col4: Gender (0:Male, 1: Female) Col5: Activity (1:sport, 2:choir, 3:other) Col6: Age
metadata2=np.loadtxt("D:\\Domonkos\\resting_raw\\metadata\\labels.txt")


for k in range(10):
     
    graph=tf.Graph()


    with graph.as_default():
        
        #Input data
        tf_train_dataset = tf.placeholder(tf.float32, name = "train_data", shape=(batch_size1, img_size_x, img_size_y, img_size_z, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, name = "train_labels", shape=(batch_size1, num_labels))
        tf_keep_prob =  tf.placeholder(tf.float32, name = "keep_prob")
    
        
        #Variables
        layer1_weights = tf.Variable(tf.truncated_normal(
            [kernel_size_1, kernel_size_1, kernel_size_1, num_channels, depth_1], stddev=0.1), name="layer1_weights")
        layer1_biases = tf.Variable(tf.zeros([depth_1]), name ="layer1_biases")
        
        layer2_weights = tf.Variable(tf.truncated_normal(
            [kernel_size_1, kernel_size_1, kernel_size_1, depth_1, depth_1], stddev=0.1), name="layer2_weights")
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth_1]), name="layer2_biases")
        
        layer3_weights = tf.Variable(tf.truncated_normal(
            [(img_size_x // 4) * (img_size_y // 4) * (img_size_z // 4 ) * depth_1, num_hid_1], stddev=0.1), name = "layer3_weights")
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hid_1]), name = "layer3_biases")
        
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hid_1, num_labels], stddev=0.1), name = "layer4_weights")
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name = "layer4_biases")
    
    
        #model
        def model(data):
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
            
            fully_connect_1_drop = tf.nn.dropout(fully_connect_1, 1)
    
            return tf.matmul(fully_connect_1_drop, layer4_weights) + layer4_biases    
    
        #computation of loss
        
        logits = model(tf_train_dataset)
        #regulizers=tf.nn.l2_loss(layer3_weights)+tf.nn.l2_loss(layer4_weights)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) #+ beta*regulizers)
    
        #optimizer
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        #predictions
        train_prediction = tf.nn.softmax(logits)
    
    
    
    with tf.Session(graph=graph) as session:
        #initializing graph
        tf.global_variables_initializer().run()
        print('Initialized \n')
    
        #shuffleing training indices
        train_index=np.arange(92)
        np.random.shuffle(train_index)
        
        #iterating over optimization steps
        for step in range(num_steps1):
            #building up the batch dataset
            offset = (step * batch_size1) % (num_samples1 - batch_size1)
            batch_data=[]
            batch_labels=[]
            for i in range(batch_size1):
                ind=train_index[(offset+i) % train_index.shape[0]]
                batch_data.append(np.load("D:\\Domonkos\\transfer_learning\\numpys_tf_use2\\" + str(ind) + "_" + str(int(np.floor(rn.random() * metadata[ind, 6]))) + ".npy"))
                batch_labels.append(metadata[ind,2])
            batch_data = np.asarray(batch_data)
            batch_labels = np.asarray(batch_labels)
            batch_labels = (np.arange(num_labels) == batch_labels[:,None]).astype(np.float32)
            #creating the variables of the graph and running the training
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels,  tf_keep_prob : keep_prob1}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            
            file.write(str(time.time()-start) + " " + str(l) + " " + str(accuracy(predictions, batch_labels)) + " " + " " + str(step) + " " + '\n')
            
            

        
        layer1_weights_tf= session.run("layer1_weights:0")
        layer1_biases_tf= session.run("layer1_biases:0")
        layer2_weights_tf= session.run("layer2_weights:0")
        layer2_biases_tf= session.run("layer2_biases:0")   
        layer3_weights_tf= session.run("layer3_weights:0")
        layer3_biases_tf= session.run("layer3_biases:0")
        layer4_weights_tf= session.run("layer4_weights:0")
        layer4_biases_tf= session.run("layer4_biases:0")
        
        print("variables are saved_" + str(k) + "\n")
    
    
    
    
    labels=np.asarray(range(num_class)) #labels for classification
    
    
    batch_size2 = 4  #number of elemnets in a batch
    num_samples2 = 60  #number of samples
        
    
    
    graph=tf.Graph()
    
    
    with graph.as_default():
        
        #Input data
        tf_train_dataset = tf.placeholder(tf.float32, name = "train_data", shape=(batch_size2, img_size_x, img_size_y, img_size_z, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, name = "train_labels", shape=(batch_size2, num_labels))
        tf_keep_prob =  tf.placeholder(tf.float32, name = "keep_prob")
        tf_valid_dataset = tf.placeholder(tf.float32, name="valid", shape=((num_samples2 // num_kmeans), img_size_x, img_size_y, img_size_z, num_channels))
        layer1_weights = tf.placeholder(tf.float32, name = "layer1_weights" , shape=(kernel_size_1, kernel_size_1, kernel_size_1, num_channels, depth_1))
        layer1_biases = tf.placeholder(tf.float32, name = "layer1_biases" , shape=(depth_1))
        layer2_weights = tf.placeholder(tf.float32, name = "layer2_weights" , shape=(kernel_size_1, kernel_size_1, kernel_size_1, depth_1, depth_1))
        layer2_biases = tf.placeholder(tf.float32, name = "layer2_biases" , shape=(depth_1))

        
        #Variables
        layer3_weights = tf.Variable(tf.truncated_normal(
            [(img_size_x // 4) * (img_size_y // 4) * (img_size_z // 4 ) * depth_1, num_hid_1], stddev=0.1), name = "layer3_weights")
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hid_1]), name = "layer3_biases")
        
        layer4_weights = tf.Variable(tf.truncated_normal(
            [num_hid_1, num_labels], stddev=0.1), name = "layer4_weights")
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]), name = "layer4_biases")
    #
        #model
        def model(data):
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
            
            fully_connect_1_drop = tf.nn.dropout(fully_connect_1, 1)
    
            return tf.matmul(fully_connect_1_drop, layer4_weights) + layer4_biases    
    
        #computation of loss
        
        logits = model(tf_train_dataset)
    
        #regulizers=tf.nn.l2_loss(layer3_weights)+tf.nn.l2_loss(layer4_weights)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) #+ beta*regulizers)
    
        #optimizer
        optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        #predictions
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    
        

    
    #running the training and computing the loss on the test data afterwards
    
    #Splitting data of the patient IDs to train and validation sets for cross validation
    kfold=KFold(num_samples2, num_kmeans, True, None)


    ind1 = 0
    for train_index, test_index  in kfold:
        np.random.shuffle(train_index)
       
        
        #building up the validation dataset ATTENTION iteration is going backwards from the end of the test_index list for the data and for the laabels as weel so they are consistent
        valid_dataset=[]
        valid_labels=[]
        for j in range(test_index.shape[0]):
            valid_dataset.append(np.load("D:\\Domonkos\\resting_raw\\numpys_resamp_ss\\rss_" + str(test_index[-(j+1)]) + "_" + str(int(np.floor(rn.random() * 100))) + ".npy"))
            valid_labels.append(metadata2[test_index[-(j+1)],2])
        valid_dataset = np.asarray(valid_dataset)
        valid_labels = np.asarray(valid_labels)
        valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
    
        with tf.Session(graph=graph) as session:
            #initializing graph
            tf.global_variables_initializer().run()
            print('Initialized globally \n')
            session.run(layer3_weights, feed_dict={layer3_weights: layer3_weights_tf})
            session.run(layer3_biases, feed_dict={layer3_biases: layer3_biases_tf})
            session.run(layer4_weights, feed_dict={layer4_weights: layer4_weights_tf})
            session.run(layer4_biases, feed_dict={layer4_biases: layer4_biases_tf})
            print('Initialized weights and biases_' + str(k) + "\n")
            
            
            #iterating over optimization steps
            for step in range(num_steps2):
                #building up the batch dataset
                offset = (step * batch_size2) % ((num_samples2 - num_samples2/num_kmeans) - batch_size2)
                batch_data=[]
                batch_labels=[]
                for i in range(batch_size2):
                    ind=train_index[(offset+i) % train_index.shape[0]]
                    batch_data.append(np.load("D:\\Domonkos\\resting_raw\\numpys_resamp_ss\\rss_" + str(ind) + "_" + str(int(np.floor(rn.random() * 100))) + ".npy"))
                    batch_labels.append(metadata2[ind,2])
                batch_data = np.asarray(batch_data)
                batch_labels = np.asarray(batch_labels)
                batch_labels = (np.arange(num_labels) == batch_labels[:,None]).astype(np.float32)
                #creating the variables of the graph and running the training
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_valid_dataset : valid_dataset, tf_keep_prob : keep_prob2, layer1_weights : layer1_weights_tf, layer1_biases : layer1_biases_tf, layer2_weights : layer2_weights_tf, layer2_biases : layer2_biases_tf}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                
                
                file2.write(str(time.time()-start) + " " + str(l) + " " + str(accuracy(predictions, batch_labels)) + " " + str(accuracy(valid_prediction.eval(feed_dict = feed_dict ), valid_labels)) + " " + str(ind1) + " " + str(step) + " " + '\n')
                
        ind1+=1
                    
                    

end = time.time()
file.close()
file2.write(str(end-start))
file2.close() 


