# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn
import pdb


class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss
class SniCoLoss_4(nn.Module):
    def __init__(self,s2):
        super(SniCoLoss_4, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.s2 = s2
    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs,contrast_pairs_2):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )
        IA_refinement = self.NCE(
            torch.mean(contrast_pairs_2['IA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )
        HA_refinement_2 = self.NCE(
            torch.mean(contrast_pairs_2['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )
        OA_refinement = self.NCE(
            torch.mean(contrast_pairs_2['OA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )
        HB_refinement_2= self.NCE(
            torch.mean(contrast_pairs_2['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )




        loss = HA_refinement + HB_refinement +\
               self.s2*( HA_refinement_2+OA_refinement+IA_refinement+HB_refinement_2)
        return loss

class SniCoLoss_3(nn.Module):
    def __init__(self,s2):
        super(SniCoLoss_3, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.s2 = s2
    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs,contrast_pairs_2):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )
        HA_refinement_2 = self.NCE(
            torch.mean(contrast_pairs_2['HA'], 1),
            torch.mean(contrast_pairs_2['EA'], 1),
            contrast_pairs_2['EB']
        )



        loss = HA_refinement + HB_refinement +self.s2*( HA_refinement_2)
        return loss

class SniCoLoss_2(nn.Module):
    def __init__(self,s2):
        super(SniCoLoss_2, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.s2 = s2
    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs,contrast_pairs_2):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )
        HA_refinement_2 = self.NCE(
            torch.mean(contrast_pairs_2['HA'], 1),
            torch.mean(contrast_pairs_2['EA'], 1),
            contrast_pairs_2['EB']
        )
        HB_refinement_2 = self.NCE(
            torch.mean(contrast_pairs_2['HB'], 1),
            torch.mean(contrast_pairs_2['EB'], 1),
            contrast_pairs_2['EA']
        )



        loss = HA_refinement + HB_refinement +self.s2*( HA_refinement_2+HB_refinement_2)
        return loss



class ActELoss_v3(nn.Module):
    def __init__(self,cfg):
        super(ActELoss_v3, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.sigma = cfg.SIGMA
        self.e_theta = cfg.E_THETA
        self.e_alpha = cfg.E_ALPHA
        self.e_g = cfg.E_G

    def forward(self, actioness,actioness_2):
        loss = 0

        actioness_0 = actioness
        actioness_3 = torch.zeros((actioness_2.shape[0], actioness_2.shape[1] + 11)).cuda()
        actioness_3[:, :6] = actioness_2[:, 0].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_3[:, -6:] = actioness_2[:, -1].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_3[:, 6:-5] = actioness_2
        actioness_4 = torch.zeros((actioness_0.shape[0], actioness_0.shape[1] + 11)).cuda()
        actioness_4[:, :6] = actioness_0[:, 0].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_4[:, -6:] = actioness_0[:, -1].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_4[:, 6:-5] = actioness_0

      #  for b in range(actioness_2.shape[0]):
        for i in range(750):
            for j in range(11):
                max_w,_ = torch.max((actioness[:,i].repeat(1, 11).reshape(actioness.shape[0],11)-actioness_4[:,i:i+11]),dim=1)
             #   print(max_w.shape)
                w_ij = torch.exp(-(torch.nn.ReLU()
                    (torch.norm(actioness_0[:,i]-actioness_4.detach()[:,i+j],p=2,dim=0)*torch.norm(actioness_0[:,i]-actioness_4.detach()[:,i+j],p=2,dim=0)
                     -self.e_g*max_w*max_w)+self.e_g*max_w*max_w)/2/self.sigma/self.sigma)

                loss =loss+(w_ij*torch.abs(actioness_2[:,i]-actioness_3[:,i+j])).mean()

        loss = loss+(self.e_theta*torch.norm(actioness-actioness_2,p=2,dim=1)*(torch.norm(actioness-actioness_2,p=2,dim=1))).mean()
        loss = loss*self.e_alpha


        return loss

class ActELoss_v2(nn.Module):
    def __init__(self,cfg):
        super(ActELoss_v2, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.sigma = cfg.SIGMA
        self.e_theta = cfg.E_THETA
        self.e_alpha = cfg.E_ALPHA

    def forward(self, actioness,actioness_2):
        loss = 0

        actioness_0 = actioness
        actioness_3 = torch.zeros((actioness_2.shape[0], actioness_2.shape[1] + 11)).cuda()
        actioness_3[:, :6] = actioness_2[:, 0].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_3[:, -6:] = actioness_2[:, -1].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_3[:, 6:-5] = actioness_2
        actioness_4 = torch.zeros((actioness_0.shape[0], actioness_0.shape[1] + 11)).cuda()
        actioness_4[:, :6] = actioness_0[:, 0].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_4[:, -6:] = actioness_0[:, -1].repeat(1, 6).reshape(actioness_2.shape[0], 6)
        actioness_4[:, 6:-5] = actioness_0

      #  for b in range(actioness_2.shape[0]):
        for i in range(750):
            for j in range(11):
                w_ij = torch.exp(-(torch.norm(actioness_0[:,i]-actioness_4.detach()[:,i+j],p=2,dim=0)*torch.norm(actioness_0[:,i]-actioness_4.detach()[:,i+j],p=2,dim=0)/2/self.sigma/self.sigma))
                loss =loss+(w_ij*torch.abs(actioness_2[:,i]-actioness_3.detach()[:,i+j])).mean()

        loss = loss+(self.e_theta*torch.norm(actioness-actioness_2,p=2,dim=1)*(torch.norm(actioness-actioness_2,p=2,dim=1))).mean()

        loss = loss*self.e_alpha


        return loss



class ActELoss(nn.Module):
    def __init__(self,cfg):
        super(ActELoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.sigma = cfg.SIGMA
        self.e_theta = cfg.E_THETA
        self.e_alpha = cfg.E_ALPHA

    def forward(self, actioness,actioness_2):
        loss = 0
        actioness_0 = actioness
        actioness_3 = torch.zeros((actioness_2.shape[0],actioness_2.shape[1]+11)).cuda()
        actioness_3[:,:6] = actioness_2[:,0].repeat(1,6).reshape(actioness_2.shape[0],6)
        actioness_3[:, -6:] = actioness_2[:,-1].repeat(1, 6).reshape(actioness_2.shape[0],6)
        actioness_3[:,6:-5] = actioness_2
        actioness_4 = torch.zeros((actioness_0.shape[0], actioness_0.shape[1] + 11)).cuda()
        actioness_4[:, :6] = actioness_0[:, 0].repeat(1, 6).reshape(actioness_2.shape[0],6)
        actioness_4[:, -6:] = actioness_0[:, -1].repeat(1, 6).reshape(actioness_2.shape[0],6)
        actioness_4[:, 6:-5] = actioness_0

      #  for b in range(actioness_2.shape[0]):
        for i in range(750):
            for j in range(11):
                w_ij = torch.exp(-torch.abs(actioness_0[:,i]-actioness_4[:,i+j])/2/self.sigma/self.sigma)
                loss =loss+(w_ij*torch.abs(actioness_2[:,i]-actioness_3[:,i+j])).sum()

        loss = loss+self.e_theta*torch.norm(actioness_0-actioness_2,p=2,dim=1).sum()
        loss = loss*self.e_alpha


        return loss



class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA']
        )


        loss = HA_refinement + HB_refinement
        return loss

class RelationLoss(nn.Module):
    def __init__(self):
        super(RelationLoss, self).__init__()
       # self.ce_criterion =


    def forward(self, re_s_scores,re_d_scores):
        loss = 0
        for s_scores in re_s_scores:
            loss+=torch.mean((1-s_scores)*(1-s_scores))/\
                  (len(re_s_scores))
        for d_scores in re_d_scores:
            loss += torch.mean(d_scores*d_scores)/\
                    (len(re_d_scores))

        return loss


class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss2(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss2, self).__init__()
        self.action_criterion = ActionLoss()
        self.relation_loss= RelationLoss()
        self.sinco_loss = SniCoLoss()
        self.theta = cfg.THETA

    def forward(self, video_scores, label, re_s_scores,re_d_scores):
        loss_cls = self.action_criterion(video_scores, label)
        loss_relation = self.relation_loss(re_s_scores,re_d_scores)
        loss_total = loss_cls + self.theta * loss_relation#+0.01*loss_sinco

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss3(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss3, self).__init__()
        self.action_criterion = ActionLoss()
        self.relation_loss= RelationLoss()
        self.snico_criterion = SniCoLoss()

        self.theta = cfg.THETA

    def forward(self, video_scores, label, re_s_scores,re_d_scores,contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_relation = self.relation_loss(re_s_scores,re_d_scores)
        loss_snico = self.snico_criterion(contrast_pairs)

        loss_total = loss_cls + self.theta * loss_relation+0.01*loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss4(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss4, self).__init__()
        self.action_criterion = ActionLoss()
        self.relation_loss= RelationLoss()
        self.snico_criterion = SniCoLoss()

        self.theta = cfg.THETA

    def forward(self, video_scores, label, re_s_scores,re_d_scores,contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
      #  loss_relation = self.relation_loss(re_s_scores,re_d_scores)
        loss_snico = self.snico_criterion(contrast_pairs)

        loss_total = loss_cls +0.01*loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_5(nn.Module):
    def __init__(self,s2):
        super(TotalLoss_5, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss_2(s2)

    def forward(self, video_scores, label, contrast_pairs,contrast_pairs_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs,contrast_pairs_2)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_6(nn.Module):
    def __init__(self,s2):
        super(TotalLoss_6, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss_3(s2)

    def forward(self, video_scores, label, contrast_pairs,contrast_pairs_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs,contrast_pairs_2)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_7(nn.Module):
    def __init__(self,s2):
        super(TotalLoss_7, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss_4(s2)

    def forward(self, video_scores, label, contrast_pairs,contrast_pairs_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs,contrast_pairs_2)
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_8(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_8, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.gama = cfg.GAMA

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + self.gama * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict



class TotalLoss_9(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_9, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss(cfg)

    def forward(self, video_scores, label, contrast_pairs,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)

        loss_total = loss_cls + 0.01 * loss_snico +loss_act
        #print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict



class TotalLoss_10(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_10, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v2(cfg)

    def forward(self, video_scores, label, contrast_pairs,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_11(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_11, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v3(cfg)

    def forward(self, video_scores, label, contrast_pairs,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act
    #    print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict


class TotalLoss_12(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_12, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v2(cfg)
        self.relation_loss = RelationLoss()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class TotalLoss_13(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_13, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v3(cfg)
        self.relation_loss = RelationLoss()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class RelationLoss_att(nn.Module):
    def __init__(self):
        super(RelationLoss_att, self).__init__()
       # self.ce_criterion =


    def forward(self, re_s_scores,re_d_scores,s_action_scores,d_action_scores):
        loss = 0
        for i,s_scores in enumerate(re_s_scores):
            loss+=torch.mean(s_action_scores[i]*(1-s_scores)*(1-s_scores))/\
                  (len(re_s_scores)+len(re_d_scores))
        for i,d_scores in enumerate(re_d_scores):
            loss += torch.mean(d_action_scores[i]*d_scores*d_scores)/\
                    (len(re_s_scores)+len(re_d_scores))

        return loss

class TotalLoss_14(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_14, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v2(cfg)
        self.relation_loss = RelationLoss_att()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2,s_attention_scores,d_attention_scores):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores,s_attention_scores,d_attention_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class SniCoLoss_v5(nn.Module):
    def __init__(self):
        super(SniCoLoss_v5, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, attention,T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
     #   print(attention.shape,torch.einsum('nc,nck->nk', [q, neg]).shape)
        l_neg = attention*(torch.einsum('nc,nck->nk', [q, neg]))
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
    #    print(l_pos.shape,l_neg.shape,logits.shape, loss.shape)
    #    pdb.set_trace()

        return loss

    def forward(self, contrast_pairs,attention):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),
            torch.mean(contrast_pairs['EA'], 1),
            contrast_pairs['EB'],
            attention[:,:150,0]

        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1),
            torch.mean(contrast_pairs['EB'], 1),
            contrast_pairs['EA'],
        attention[:,150:300,0]
        )


        loss = HA_refinement + HB_refinement
        return loss


class TotalLoss_15(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_15, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss_v5()
        self.act_eneragy_criterion = ActELoss_v2(cfg)
        self.relation_loss = RelationLoss()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2,attention_scores):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs,attention_scores)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict


class TotalLoss_16(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_16, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v2(cfg)
        self.gama = cfg.GAMA

    def forward(self, video_scores, label, contrast_pairs,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + self.gama * loss_snico +loss_act
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict

class RelationLoss_v2(nn.Module):
    def __init__(self):
        super(RelationLoss_v2, self).__init__()
        self.ce_criterion = nn.BCELoss()


    def forward(self, re_s_scores,re_d_scores):
        loss = 0
        labels_z = torch.zeros(re_d_scores[0].shape[0], dtype=torch.float).cuda().unsqueeze(-1)
        labels_o = torch.ones(re_s_scores[0].shape[0], dtype=torch.float).cuda().unsqueeze(-1)

        for s_scores in re_s_scores:
         #   print(s_scores,'----')
            loss=loss+self.ce_criterion(s_scores,labels_o)/\
                  (len(re_s_scores))
        for d_scores in re_d_scores:
        #    print(d_scores, '++++')
            loss =loss+ self.ce_criterion(d_scores,labels_z)/\
                    (len(re_d_scores))

        return loss

class TotalLoss_17(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_17, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v2(cfg)
        self.relation_loss = RelationLoss_v2()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
        #print(loss_relation)
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict


class TotalLoss_18(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_18, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.relation_loss = RelationLoss_v2()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)

        loss_total = loss_cls + 0.01 * loss_snico +self.theta*loss_relation
        #print(loss_relation)
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict


class TotalLoss_19(nn.Module):
    def __init__(self,cfg):
        super(TotalLoss_19, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()
        self.act_eneragy_criterion = ActELoss_v3(cfg)
        self.relation_loss = RelationLoss_v2()
        self.theta = cfg.THETA


    def forward(self, video_scores, label, contrast_pairs,re_s_scores, re_d_scores,actioness,actioness_2):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_relation = self.relation_loss(re_s_scores, re_d_scores)
        loss_act = self.act_eneragy_criterion(actioness.detach(),actioness_2)
        loss_total = loss_cls + 0.01 * loss_snico +loss_act+self.theta*loss_relation
        #print(loss_relation)
      #  print('actE',float(loss_act),'Total_loss',float(loss_total))
        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/ActE': loss_act,
            'Loss/Relation': loss_relation,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict
