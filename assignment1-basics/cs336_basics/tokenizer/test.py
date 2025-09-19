
from collections import Counter
from typing import List , Tuple , Iterable



def update_cor(best_pair, cor, pair_counts, word_freq):
    """
    在一个词的token序列（cor）中合并best_pair，更新全局的pair_counts，并返回新的序列。
    """
    # 1. 统计合并前此序列中的pair数量
    old_stats = Counter(zip(cor, cor[1:]))

    # 2. 执行合并，生成新序列
    new_cor = []
    i = 0
    merged_token = best_pair[0] + best_pair[1]
    while i < len(cor):
        # 检查当前位置是否可以形成best_pair
        if i < len(cor) - 1 and (cor[i], cor[i + 1]) == best_pair:
            new_cor.append(merged_token)
            i += 2  # 跳过两个已经合并的token
        else:
            new_cor.append(cor[i])
            i += 1

    # 3. 统计合并后新序列中的pair数量
    new_stats = Counter(zip(new_cor, new_cor[1:]))

    # 4. 手动计算变化量，因为Counter的减法会忽略负数结果
    #    合并所有涉及到的pair的键
    all_involved_pairs = old_stats.keys() | new_stats.keys()

    for pair in all_involved_pairs:
        # 计算这个pair在此次合并中的净变化次数
        change = new_stats.get(pair, 0) - old_stats.get(pair, 0)
        
        if change != 0:
            # 将总变化量 (change * word_freq) 应用到全局计数上
            pair_counts[pair] = pair_counts.get(pair, 0) + change * word_freq

    return new_cor

best_pair = ('a','b')
cor =['a','b','a','b','a','b']
word_freq = 10 
pair_counts = {('a','b'):30,('b','a'):20}

print(update_cor(best_pair,cor,pair_counts,word_freq))
print(pair_counts)