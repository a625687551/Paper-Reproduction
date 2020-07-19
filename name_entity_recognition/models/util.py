# -*- coding: utf-8 -*-
# @Time    : 2020-07-18 16:35
# @Author  : wangjianfeng
# @File    : util.

"""
工具箱子
"""

import torch
import torch.nn.functional as F


# -----------------CRF工具----------------------
def word2features(sent, i):
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s>" if i == (len(sent - 1)) else sent[i + 1]
    features = {
        "w": word,
        "w-1": prev_word,
        "w+1": next_word,
        "w-1:w": prev_word + word,
        "w:w+1": word + next_word,
        "bias": 1,
    }
    return features


def sent2features(sent):
    """抽取特征序列"""
    return [word2features(sent, i) for i in range(len(sent))]


# -----------------LSTM工具----------------------

def tensorized(batch, maps):
    PAD = maps.get("<pad>")
    UNK = maps.get("<unk>")

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素长度
    lengths = [len(l) for l in batch]
    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    paris = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(paris)), key=lambda k: len(paris[k][0]),
                     reverse=True)
    paris = [paris[i] for i in indices]
    word_lists, tag_lists = list(zip(*paris))
    return word_lists, tag_lists, indices


def indexed(targets, tagset_size, start_id):
    """
    将targets中的数转化为在[T*T]大小序列中的索引，T是标注的种类
    :param targets:
    :param tagset_size:
    :param start_id:
    :return:
    """
    batch_size, max_len = targets.size()
    for col in range(max_len - 1, 0, -1):
        targets[:, col] += (targets[:, col - 1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


def cal_loss(logits, targets, tag2id):
    """
    损失计算
    :param logits: [B, L, out_size]
    :param targets: [B, L]
    :param tag2id: [B]
    :return:
    """
    PAD = tag2id.get("<pad>")
    assert PAD is not None

    mask = (targets != PAD)
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.mask_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)
    assert logits.size(0) == targets.size(0)

    loss = F.cross_entropy(logits, targets)

    return loss


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """
    计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    :param crf_scores:
    :param targets:
    :param tag2id:
    :return:
    """
    pad_id = tag2id.get("<pad>")
    start_id = tag2id.get("<start>")
    end_id = tag2id.get("<end>")

    device = crf_scores.device
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # compute golden scores method 1
    targets = targets.masked_select(mask)
    flatten_scores = crf_scores.mask_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size * target_size).contiguous()

    golden_socres = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)
    ).sum()

    # compute golden scores method 2
    # --------省略

    # compute all path scores
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t, t,
                                           start_id, :]
        else:
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()
    # 训练两个epoch loss变成负数
    loss = (all_path_scores - golden_socres) / batch_size
    return loss
