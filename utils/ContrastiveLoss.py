import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F


    
class ContrastiveLoss(nn.Module):
    def __init__(self, cls_num, feature_num):
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num

        #随机10个anchor
        if cls_num > feature_num:
            self.anchor = nn.Parameter((torch.randn(cls_num, feature_num)), requires_grad=True)
        else:
            I = torch.eye(feature_num,feature_num)
            index = torch.LongTensor(random.sample(range(feature_num), cls_num))
            init = torch.index_select(I, 0, index)
            # for i in range(cls_num):
            #     if i % 2 == 0:
            #         init[i] = -init[i]
            self.anchor = nn.Parameter(init, requires_grad=True)    
        
        
    def forward(self, feature, _target):

        centre = self.anchor.cuda()

        #print(feature.size())
        scores_matrix = torch.mm(feature, centre.T)				
        exp_scores_matrix = torch.exp(scores_matrix)		
        #print(exp_scores_matrix.size())
        total = torch.sum(exp_scores_matrix, dim=1)			
        loss_vector = torch.zeros_like(total)
        
        for index, value in enumerate(_target):
            #print(index, value)
            # if exp_scores_matrix[index][value.long()] < 0.05:
            #     print("toal", exp_scores_matrix[index][value.long()])
            loss_vector[index] = -torch.log(exp_scores_matrix[index][value.long()]/total[index])
        #print(loss_vector)
        
        loss = torch.mean(loss_vector)
   
        return loss