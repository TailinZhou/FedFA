import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F



class AnchorLoss(nn.Module):
    def __init__(self, cls_num, feature_num, ablation=0):
        """
        :param cls_num: class number
        :param feature_num: feature dimens
        """
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num

        # initiate anchors
        if cls_num > feature_num:
            self.anchor = nn.Parameter(F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True)
        elif ablation==1:
            self.anchor = nn.Parameter(F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True)
        elif ablation==2:
            self.anchor = nn.Parameter(F.normalize(torch.randn(cls_num, feature_num)), requires_grad=True)
            self.anchor.data = torch.load('utils/converged_anchors_data.pt')
        else:
            I = torch.eye(feature_num,feature_num)
            index = torch.LongTensor(random.sample(range(feature_num), cls_num))
            init = torch.index_select(I, 0, index)
            # for i in range(cls_num):
            #     if i % 2 == 0:
            #         init[i] = -init[i]
            self.anchor = nn.Parameter(init, requires_grad=True)

        
        
    def forward(self, feature, _target, Lambda = 0.1):
        """
        :param feature: input
        :param _target: label/targets
        :return: anchor loss 
        """
        # broadcast feature anchors for all inputs
        centre = self.anchor.cuda().index_select(dim=0, index=_target.long())
        # compute the number of samples in each class
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num-1)
        count = counter[_target.long()]
        centre_dis = feature - centre				# compute distance between input and anchors
        pow_ = torch.pow(centre_dis, 2)				# squre
        sum_1 = torch.sum(pow_, dim=1)				# sum all distance
        dis_ = torch.div(sum_1, count.float())		# mean by class
        sum_2 = torch.sum(dis_)/self.cls_num						# mean loss
        res = Lambda*sum_2   							# time hyperparameter lambda 
        return res
    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, cls_num, feature_num):
        """
        :param cls_num: class number
        :param feature_num: feature dimens
        """
        super().__init__()
        self.cls_num = cls_num
        self.feature_num = feature_num

        # initiate anchors
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
        counter = torch.histc(_target, bins=self.cls_num, min=0, max=self.cls_num-1)
        count = counter[_target.long()]
        #print(feature.size())
        scores_matrix = torch.mm(feature, centre.T)				
        exp_scores_matrix = torch.exp(scores_matrix)			
        #print(exp_scores_matrix.size())
        total = torch.sum(exp_scores_matrix, dim=1)			
        #print(total.size())
        loss_vector = torch.zeros_like(total)
        
        for index, value in enumerate(_target):
            #print(index, value)
            # if exp_scores_matrix[index][value.long()] < 0.05:
            #     print("toal", exp_scores_matrix[index][value.long()])
            loss_vector[index] = -torch.log(exp_scores_matrix[index][value.long()]/total[index])
        #print(loss_vector)
        
        loss = torch.mean(loss_vector)
        return loss