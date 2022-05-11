import os
from scipy.io import loadmat
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
import logging

#Method to Redefine Labels
def label_to_int(label):
    if "palm" in label:
        return 0
    if "spher" in label:
        return 1
    if "hook" in label:
        return 2
    if "cyl" in label:
        return 3
    if "lat" in label:
        return 4
    if "tip" in label:
        return 5

#Method To Read From Current Directory
def read_data():
    # holds intermediate data
    X = np.empty((0,2,3000))
    y = np.empty(0) 
    #30 Samples, 2 Variables, 3000 Time Steps
    X_intermediate = np.zeros((30,2,3000)) #To be 3D array of sample, variables, time step
    y_intermediate = np.empty(30) #To be Labels
    channel1 = np.empty((30,3000))
    channel2 = np.empty((30,3000))
    label = ""
    directory1 = "Database1"
    clients = []
    data_dict = {}
    nested_dictionary = {}

    #Load All Data In
    for root, subdirectories, files in os.walk(directory1):
        for file in files:
            path = os.path.join(root, file)
            if ".mat" in path:
                data = loadmat(path)
                clientid = re.search("([^.]+)", file).group(0)
                clients = np.append(clients, clientid)
                for x in data.keys():
                    if "1" in x:
                        label = re.search("([^_]+)", x).group(0)
                        label = label_to_int(label)
                        channel1 = data.get(x)
                        channel1 = (channel1-np.mean(channel1))/np.std(channel1)
                    if "2" in x:
                        channel2 = data.get(x)
                        channel2 = (channel2-np.mean(channel2))/np.std(channel2)
                        for i in range(30):
                            X_intermediate[i, :, :] = np.array([channel1[i,:], channel2[i,:]])
                            y_intermediate[i] = label
                        X = np.append(X, X_intermediate, axis=0)
                        y = np.append(y, y_intermediate, axis=0)
                nested_dictionary = {clientid: {'x' : X, 'y' :  y}}
                data_dict.update(nested_dictionary)
                X = np.empty((0,2,3000))
                y = np.empty(0)

    #Split into test and train
    train_data = {}
    test_data = {}
    for i in data_dict.keys():
        client_data = data_dict[i]
        X_train, X_test, y_train, y_test = train_test_split(client_data["x"], client_data["y"], test_size=0.3, stratify=client_data["y"])
        nested_dictionary = {i: {'x' : X_train, 'y' :  y_train}}
        train_data.update(nested_dictionary)
        nested_dictionary = {i: {'x' : X_test, 'y' :  y_test}}
        test_data.update(nested_dictionary)

    return clients, train_data, test_data

#Splits Data into Batches

def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


#Patition Data Main Method for FedML
def load_partition_data_semg(batch_size):
    clients, train_data, test_data = read_data()

    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    client_idx = 0

    logging.info("loading data...")

    for i in clients:
        user_train_data_num = len(train_data[i]['x'])
        user_test_data_num = len(test_data[i]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[i], 10)
        test_batch = batch_data(test_data[i], 10)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch
        client_idx += 1

    logging.info("finished the loading data")

    client_num = client_idx
    class_num = 6
    return client_num, train_data_num, test_data_num, train_data_global, test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

