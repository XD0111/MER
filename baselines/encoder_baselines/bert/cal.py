# -*- coding: utf-8 -*-#
import math


# -------------------------------------------------------------------------------
# Name:         cal
# Description:
# Author:       梁超
# Date:         2024/7/28
# -------------------------------------------------------------------------------
def cal(a, b, c):
    t = 0.4 * a + 0.3 * b + 0.3 * c
    print(t)
    return t


def softmax(lst):
    # 计算 e 的指数幂
    exp_lst = [math.exp(x) for x in lst]
    # 计算总和
    sum_exp = sum(exp_lst)
    # 应用 softmax 公式
    softmax_lst = [x / sum_exp for x in exp_lst]
    print(softmax_lst)
    return softmax_lst


if __name__ == '__main__':
    softmax([cal(0.35, 0.11, 0.12), cal(0.25, 0.76, 0.80), cal(0.40, 0.13, 0.08)])
