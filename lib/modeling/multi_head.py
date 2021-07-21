"""
reference: GroupFace
borrowed from: https://github.com/cybercore-co-ltd/track2_aicity_2021/blob/master/lib/modeling/multiheads_baseline.py
"""
import torch
import torch.nn as nn


class FC(nn.Module):
    def __init__(self, inplanes, outplanes, bn_func=nn.BatchNorm2d):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        self.bn = bn_func(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        return self.act(x)


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256, bn_func=nn.BatchNorm2d):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim, bn_func=bn_func)
        self.fc2 = FC(intermediate_dim, outplanes, bn_func=bn_func)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.softmax(out, dim=1)


class MultiHeads(nn.Module):
    def __init__(self, feature_dim=512, groups=4, mode='S', backbone_fc_dim=2048, bn_func=nn.BatchNorm2d):
        super(MultiHeads, self).__init__()
        self.mode = mode
        self.groups = groups
        # self.Backbone = backbone[resnet]
        self.instance_fc = FC(backbone_fc_dim, feature_dim, bn_func=bn_func)
        self.GDN = GDN(feature_dim, groups, bn_func=bn_func)
        self.group_fc = nn.ModuleList([FC(backbone_fc_dim, feature_dim, bn_func=bn_func) for i in range(groups)])
        self.feature_dim = feature_dim

    def forward(self, x):
        B = x.shape[0]
        # x = self.Backbone(x)  # (B,2048)
        instance_representation = self.instance_fc(x)  # (B,512)

        # GDN
        group_inter, group_prob = self.GDN(instance_representation)
        # print(group_prob)
        # group aware repr
        v_G = [Gk(x) for Gk in self.group_fc]  # [(B,512)] x groups

        # self distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(dim=-1).expand(self.groups, B).T) / self.groups + (
                1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data

        if self.mode == 'S':
            # group ensemble
            group_mul_p_vk = list()
            for k in range(self.groups):
                Pk = group_prob[:, k].unsqueeze(dim=-1).expand(B, self.feature_dim)
                group_mul_p_vk.append(torch.mul(v_G[k], Pk))
            group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
            # instance , group aggregation
            final = instance_representation + group_ensembled
        else:
            final = instance_representation

        return group_inter, final, group_prob, group_label
