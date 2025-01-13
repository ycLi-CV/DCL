import random

import torch
from torch import nn


# 这里给予了两种写法
# 读者都可试试
class CorrespondingLoss(nn.Module):

    def __init__(self, exp, batch_size=8):
        super(CorrespondingLoss, self).__init__()
        self.exp = exp
        self.batch_size = batch_size

    def forward(self, zero_output, one_output, labels):
        '''

        Args:
            zero_output: like the feature of [cover1, stego2, stego3, cover4]
            one_output: like the feature of [stego1, cover2, cover3, stego4]
            labels:[[0, 1], [1, 0], [1, 0], [0, 1]]
        一一对应即可
        Returns:The answer of total contrastive loss.

        '''
        zero_output = zero_output.reshape(zero_output.size(0), -1)
        one_output = one_output.reshape(one_output.size(0), -1)

        batch_size, _ = zero_output.size()
        zero = torch.zeros_like(zero_output)
        one = torch.zeros_like(one_output)

        for index, label_number in enumerate(labels[:, 0]):
            if int(label_number) == 0:
                zero[index] = zero_output[index]
                one[index] = one_output[index]
            elif int(label_number) == 1:
                zero[index] = one_output[index]
                one[index] = zero_output[index]
        # 下面那个sum类型的需要这两个
        zero_output = zero
        one_output = one

        del zero, one
        com_lis = min_combinations(zero_output.size(0))
        cover1_abs = zero_output.norm(dim=1)
        stego1_abs = one_output.norm(dim=1)

        cover1_abs = torch.where(cover1_abs == 0, torch.ones_like(cover1_abs), cover1_abs)
        stego1_abs = torch.where(stego1_abs == 0, torch.ones_like(stego1_abs), stego1_abs)
        # 上述首先先将Cover和stego分开，其中mask和com_lis
        # 表示随机取batchsize-1个数的正正或者负负样本图像
        mask = torch.zeros(batch_size, batch_size).cuda()
        for i in com_lis:
            mask[i[0], i[1]] = 1
        posc_matrix = torch.einsum(
            "ik,jk->ij", zero_output, zero_output
        ) / torch.einsum("i,j->ij", cover1_abs, cover1_abs)

        posc_matrix = torch.mul(mask, posc_matrix)

        poss_matrix = torch.einsum(
            "ik,jk->ij", one_output, one_output
        ) / torch.einsum("i,j->ij", stego1_abs, stego1_abs)

        poss_matrix = torch.mul(mask, poss_matrix)

        sim_matrix = torch.einsum(
            "ik,jk->ij", zero_output, one_output
        ) / torch.einsum("i,j->ij", cover1_abs, stego1_abs)

        loss_contrastive = -torch.mean(0.5 * posc_matrix.sum(dim=1) + 0.5 * poss_matrix.sum(dim=1) - sim_matrix[
            range(batch_size), range(batch_size)])
        # return the total loss, cc_loss, ss_loss and cs_loss
        return loss_contrastive, posc_matrix.sum(dim=1).sum(), poss_matrix.sum(dim=1).sum(), sim_matrix[
            range(batch_size), range(batch_size)].sum()



class CorrespondingLossSum(nn.Module):

    def __init__(self):
        super(CorrespondingLossSum, self).__init__()

    def forward(self, zero_output, one_output):
        # 这里和上述一样，除了写法不同外，我们这里同时计算每个图像之间的值
        # 上述是从batchsize * batchsize个数中取batchsize-1个特征图
        # 而我们这里直接计算batchsize * batchsize个特征图(covers和stegos都是这也)，总的算法还是一样的，只不过计算的特征图不同
        # 这里要使用的zero_output和one_output与上面一样，都需要分开之后的特征图
        cc_loss = cos_similar(zero_output, zero_output) * 0.5
        ss_loss = cos_similar(one_output, one_output) * 0.5
        cs_loss = cos_similar(zero_output, one_output)
        return -(0.5 * (cc_loss + ss_loss) - cs_loss).mean(), cc_loss.sum(), ss_loss.sum(), cs_loss.sum()


def min_combinations(size):
    # 顺123456最小组合
    index_list = list(range(size))
    com_list = []
    for i in range(size - 1):
        del index_list[0]
        random_choice = random.sample(index_list, 1)
        com_list.append((i, random_choice[0]))

    return com_list

def cos_similar(first, second):
    cosine_matrix = torch.nn.functional.cosine_similarity(
        first.unsqueeze(1), second.unsqueeze(0), dim=-1
    )  # 计算每对 (first[i], second[j]) 的余弦相似度

    cos_sum_tensor = cosine_matrix.sum(dim=1)

    return cos_sum_tensor