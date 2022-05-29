# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
import pdb





# (a) Feature Embedding and (b) Actionness Modeling
class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Actionness_Module, self).__init__()
        self.len_feature = len_feature
        self.f_embed = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU()
        )

        self.timefilter = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
        )
        self.timefilter = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU())
        self.dropout = nn.Dropout(p=0.7)

        self.attention_module = nn.MultiheadAttention(embed_dim=2048, num_heads=4,batch_first=True,dropout=0.5)


        self.conv = nn.Sequential(
            nn.Conv1d(2048, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 32, 3, padding=1), nn.LeakyReLU(0.2),
            # nn.Conv1d(512, 128, 3, padding=1), nn.LeakyReLU(0.2),
        )
        self.linear1 = nn.Sequential(nn.Linear(74, 10),
                                     nn.LeakyReLU(0.2),
                                     )
        self.linear2 = nn.Sequential(nn.Linear(10 * 32, 1),
                                     nn.Sigmoid())

    def forward(self, x,flag):
        if flag:
            out = x.permute(0, 2, 1)
            out = self.f_embed(out)
            embeddings = out.permute(0, 2, 1)
            out = self.dropout(out)
            out = self.f_cls(out)
            cas = out.permute(0, 2, 1)
            actionness = cas.sum(dim=2)

            actionness_2 = self.timefilter(actionness.unsqueeze(1)).squeeze(1)

            return embeddings, cas, actionness,actionness_2
        else:

           # x = torch.cat([input1, input2], dim=1).transpose(-1, -2)
            x = x.transpose(-1,-2)
            input1 = x[:,:37,:]
            input2 = x[:, 37:,:]
            attention_output_2, attention_weights_12 = self.attention_module(query=input1, key=input2, value=input2)
            attention_output_1, attention_weights_21 = self.attention_module(query=input2, key=input1, value=input1)

            x = torch.cat([attention_output_1, attention_output_2], dim=1).transpose(-1, -2)

            x = self.conv(x)

            x = self.linear1(x)
            x = x.view(x.size(0), -1)
            x = self.linear2(x)

            return x



# CoLA Pipeline
class CoLA(nn.Module):
    def __init__(self, cfg):
        super(CoLA, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES

        self.actionness_module = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES)
      #  self.relation_module = Relation_Module(cfg.FEATS_DIM)

       # self.refine_module = Refine_Module(cfg.FEATS_DIM,cfg.NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)

        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        sorted_scores, _ = cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k_easy, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores

    def split(self,y):
        y0 = y[:,:37,:]
        y1 = y[:,37:74,:]
        y2 = y[:,74:111,:]
        y3 = y[:,111:148,:]
        return y0,y1,y2,y3

    def forward(self, x,train):
        if train:
            num_segments = x.shape[1]
            k_easy = num_segments // self.r_easy
            k_hard = num_segments // self.r_hard

            embeddings, cas, actionness,actionness_2 = self.actionness_module(x,True)

            easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
            hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)

            easy_act_0,easy_act_1,easy_act_2,easy_act_3 = self.split(easy_act)
            easy_bkg_0,easy_bkg_1,easy_bkg_2,easy_bkg_3 = self.split(easy_bkg)

        #    re_a_01 = self.relation_module(easy_act_2,hard_act)
            act_32 = torch.cat([easy_act_3,  easy_act_2], dim=1).transpose(-1, -2)
            re_a_02 = self.actionness_module(act_32,False)
            act_03 = torch.cat([easy_act_0,  easy_act_3], dim=1).transpose(-1, -2)
            re_a_03 = self.actionness_module(act_03,False)
            act_1h = torch.cat([easy_act_1,  hard_act], dim=1).transpose(-1, -2)
            re_a_0h = self.actionness_module(act_1h,False)
           #re_a_01 = self.relation_module(easy_act_0,hard_act)

        #   re_b_01 = self.relation_module(easy_bkg_2, easy_bkg_1)
            bkg_32 = torch.cat([easy_bkg_3, easy_bkg_2], dim=1).transpose(-1, -2)
            re_b_02 = self.actionness_module(bkg_32, False)
            bkg_03 = torch.cat([easy_bkg_0, easy_bkg_3], dim=1).transpose(-1, -2)
            re_b_03 = self.actionness_module(bkg_03, False)
            bkg_1h = torch.cat([easy_bkg_1, hard_bkg], dim=1).transpose(-1, -2)
            re_b_0h = self.actionness_module(bkg_1h, False)

            re_ab_23 = self.actionness_module(torch.cat([easy_act_2,easy_bkg_3],dim=1).transpose(-1,-2),False)
            re_ab_32 = self.actionness_module(torch.cat([easy_bkg_2,easy_act_3],dim=1).transpose(-1,-2),False)
            re_ab_33 =  self.actionness_module(torch.cat([easy_act_3,easy_bkg_3],dim=1).transpose(-1,-2),False)
            re_ab_h3 =   self.actionness_module(torch.cat([hard_act,easy_bkg_3],dim=1).transpose(-1,-2),False)
            re_ab_3h =  self.actionness_module(torch.cat([easy_act_3,hard_bkg],dim=1).transpose(-1,-2),False)
            re_ab_hh =    self.actionness_module(torch.cat([hard_bkg,hard_act],dim=1).transpose(-1,-2),False)

            re_s_scores = [re_a_0h,re_a_02,re_a_03,re_b_0h,re_b_02,re_b_03]
            re_d_scores = [re_ab_hh,re_ab_3h,re_ab_h3,re_ab_33,re_ab_32,re_ab_23]




         #   re_b_01 = self.relation_module(easy_bkg_0, hard_bkg)


            video_scores = self.get_video_cls_scores(cas, k_easy)


            contrast_pairs = {
                'EA': easy_act,
                'EB': easy_bkg,
                'HA': hard_act,
                'HB': hard_bkg
           }

            return video_scores, actionness,actionness_2, cas,re_s_scores,re_d_scores,contrast_pairs
        else:

            num_segments = x.shape[1]
            k_easy = num_segments // self.r_easy
            k_hard = num_segments // self.r_hard

            embeddings, cas, actionness,actionness_2 = self.actionness_module(x,True)


            #   re_b_01 = self.relation_module(easy_bkg_0, hard_bkg)

            video_scores = self.get_video_cls_scores(cas, k_easy)

            # contrast_pairs = {
            #     'EA': easy_act,
            #     'EB': easy_bkg,
            #     'HA': hard_act,
            #     'HB': hard_bkg
            # }

            return video_scores, actionness, cas
