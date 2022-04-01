import torch
import torch.nn as nn
import numpy as np
# Fuse diff results to obtain a new result
class SimFusion(nn.Module):
    def __init__(self, phi):
        super(SimFusion, self).__init__()
        self.phi = phi
        # self.mark = 1e3

    def forward(self, Sim1, Sim2, Sim3):

        Sim1 = torch.from_numpy(Sim1)
        Sim2 = torch.from_numpy(Sim2)
        Sim3 = torch.from_numpy(Sim3)

        sim = torch.stack((Sim1, Sim2, Sim3), dim=1).transpose(1, 2)
        shape1, shape2, shape3 = sim.shape

        # fuse the diff results
        sim_weight = self.compute_distance(Sim1, Sim2, Sim3).reshape(shape1, shape2, shape3)

        Sim_Fusion = torch.sum(torch.mul(sim, sim_weight), dim=2)

        return np.array(Sim_Fusion)

    def compute_distance(self, s1, s2, s3):

        # The average of the similarity
        image_len, text_len = s1.shape
        average = (s1 + s2 + s3)/3.0

        diff_s1 = torch.abs(s1 - average).view(image_len, text_len, 1)
        print(torch.abs(s1-average))
        diff_s2 = torch.abs(s2 - average).view(image_len, text_len, 1)
        diff_s3 = torch.abs(s3 - average).view(image_len, text_len, 1)

        # co_diff--->[-1, 3]
        co_diff = torch.stack((diff_s1, diff_s2, diff_s3), dim=1).transpose(1, 2).reshape(-1, 3)


        # tensor ---> np.array --->list--->[[],[],...[]]
        co_diff_list = np.array(co_diff).tolist()

        # weights --->[-1, 3]
        weights = self.weight_att(co_diff_list)

        return weights

    def weight_att(self, Co_diff_list):
        '''
        return [N*5N, 3]
        '''
        lengths = len(Co_diff_list)
        # results = [[],[],[],...]
        results = []

        for i in range(lengths):
            att = self.Compute_atten(Co_diff_list[i])
            results.append(att)

        return torch.from_numpy(np.array(results))

    def Compute_atten(self, simList):
        '''
        simList: [s1, s2, s3]
        return: The weights of each similarity score---> weight_list---> [w1, w2, w3]
        '''
        maxValue = max(simList)
        max_index = simList.index(maxValue)
        simList = torch.Tensor(simList)
        if (maxValue-self.phi) > 0:
            expList = torch.exp(simList)
            sum_expList = torch.sum(expList)
            increase_Value = torch.exp(torch.tensor(maxValue))
            weight = (expList/(sum_expList)).numpy().tolist()
            # increase_Value = torch.exp(torch.tensor(maxValue))
            # weight = (expList / (sum_expList - increase_Value)).numpy().tolist()
            weight[max_index] = 0.0

            return weight

        else:
            simList = torch.Tensor(simList)
            expList = torch.exp(simList)
            sum_List = torch.sum(expList)
            weight = (expList/sum_List).numpy().tolist()

            return weight












