#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np


def train_loss_show(args, loss_dict, clients_index):
    fig, ax = plt.subplots() 
    fig1, ax1 = plt.subplots() 
    global_loss = 0.0
    global_loss_list = []

    for i in range(args.r):
        for j in range(args.E):

            for k in (clients_index[i]):
                epoch = args.E*i+j+1
                global_loss += loss_dict[k][epoch]
        
            global_loss = global_loss/len(clients_index[i])
            global_loss_list.append(global_loss)

        
    
    for k in range(args.K):
        E_list = [i+1 for i in range(args.r * args.E)]
        # if args.verbose:
        #     print(E_list, loss_dict[k])
        ax.plot(E_list, loss_dict[k][1:])
        #ax.plot(E_list, loss_dict[k][1:], label=('client %d' % k))
        ax.set_title('Train_Loss vs. Epoch')
        ax.set_ylabel('Train_Loss')
        ax.set_xlabel('Epoch') 
        #ax.legend()
           
    ax1.plot(E_list, global_loss_list, label=('global'))
    ax1.set_title('Train_Loss vs. Epoch')
    ax1.set_ylabel('Global Train_Loss')
    ax1.set_xlabel('Epoch') 
    ax1.legend()
 

    plt.show() 
    
    
def train_localacc_show(args, localacc_list):
    fig1, ax1 = plt.subplots() 
    E_list = [i+1 for i in range(args.r)]       
    ax1.plot(E_list, localacc_list, label=('mean local accuracy'))
    ax1.set_title('mean local accuracy vs. round')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('round') 
    ax1.legend()
 

    plt.show() 

