import os
from turtle import up
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter, defaultdict
from typing import List , Tuple , Iterable


class BPE:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def train(self, input_path: str | os.PathLike) -> None:
        self.merges = []
        self.vocab = {}



        # 基础词表
        byte_to_unicode_map = gpt2_bytes_to_unicode()
        unicode_to_byte_map = {v: k for k, v in byte_to_unicode_map.items()}
        sorted_unicode_chars = sorted(unicode_to_byte_map.keys())
        for unicode_char in sorted_unicode_chars:
            original_byte = unicode_to_byte_map[unicode_char]
            self.vocab[len(self.vocab)] =bytes([original_byte])
        next_id = 256
        # 遍历特殊词表，用于跳过
        for token in self.special_tokens:
            self.vocab[next_id] = token.encode("utf-8")
            next_id += 1
        
        # 增量更新模式
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

        def update(corpus, best_pair,word_counter,word_map,pair_counts):
            best_str = best_pair[0] + best_pair[1]
            for word in word_counter: 
                if best_str not in word:
                    continue

                cor = corpus[word_map[word]]
                word_freq = word_counter[word]
                new_cor = update_cor(best_pair,cor,pair_counts,word_freq)


                # last_flag = True
                # i = 0
                # while i < len(cor) - 1:
                #     if cor[i] == best_pair[0] and cor[i+1] == best_pair[1]:
                #         if i > 0:
                #             pre_token = cor[i-1]
                #             pair = (pre_token, best_pair[0])
                #             pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                #             pair = (pre_token, best_pair[0] + best_pair[1])
                #             pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]

                #         if i+2 < len(cor):
                #             next_token = cor[i+2]
                #             pair = (best_pair[1], next_token)
                #             pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                #             pair = (best_pair[0] + best_pair[1], next_token)
                #             pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]
    
                #         pair = (best_pair[0], best_pair[1])
                #         pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                #         new_cor.append(best_str)
                #         if i+1 == len(cor) - 1:
                #             last_flag = False
                #         i += 2
                #     else:
                #         new_cor.append(cor[i])
                #         i += 1
                # if last_flag:
                #     new_cor.append(cor[-1])
                # corpus[word_map[word]] = new_cor
             
            pair_counts = {k: v for k, v in pair_counts.items() if v > 0}

        def to_bytes_tuple(word: str) -> Tuple[bytes]:
            l = list(tuple(word.encode("utf-8")))
            l = [bytes([x]) for x in l]
            return tuple(l)
        # 开始训练
        with open(input_path, "rb") as f:
            # 分块
            # boundaries = find_chunk_boundaries(
            #     f, 1, "<|endoftext|>".encode("utf-8"))
            # boundary_start, boundary_end = boundaries[0], boundaries[1]
            # chunk = f.read(boundary_end - boundary_start).decode("utf-8", errors="ignore")

            # escaped_tokens = [re.escape(token) for token in self.special_tokens]
            # special_tokens_pattern = r'|'.join(escaped_tokens)

            # word_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            # full_pattern = f"{special_tokens_pattern}|{word_pattern}"
            # words = [match.group(0) for match in re.finditer(full_pattern, chunk)]

            boundaries = find_chunk_boundaries(
                f, 1, "<|endoftext|>".encode("utf-8"))
            boundary_start, boundary_end = boundaries[0], boundaries[1]
            text = f.read(boundary_end - boundary_start).decode("utf-8", errors="ignore")
            
            # 去除special token
            chunks = re.split("|".join(map(re.escape, self.special_tokens)), text)

            words = []
            pre_tokens_cnt = defaultdict(int)
            for chunk in chunks:
                for m in re.finditer(self.PAT, chunk):
                    word = m.group(0)
                    word_tuple = to_bytes_tuple(word)
                    pre_tokens_cnt[word_tuple] += 1
                
            # 预分词优化
            # word_counter = Counter(words)

            # corpus = [] # word 阶段融合词表 [w,o,r,d] -> [wo,r,d]-> [wor,d] ->[word]
            # token_map = {}

    
            # for word in word_counter:
            #     path =[]
            #     for i in range(len(word)):
            #         path.append(word[i])
            #     corpus.append(path)
            #     token_map[word] = len(corpus) - 1

            # # print(corpus)
            # # print(token_map)

            # pair_counts = {}

            # # 预分词
            # for word in word_counter:
            #     word_path = corpus[token_map[word]]
            #     for i in range(len(word_path) - 1):
            #         pair = (word_path[i], word_path[i+1])
            #         pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]



            
            while len(self.vocab) < self.vocab_size :
                pair_count = defaultdict(int)

                # 统计频次
            
                for tokens , cnt  in pre_tokens_cnt.items():
                    for i in range(len(tokens)-1):
                        pair_count[(tokens[i],tokens[i+1])] += 1
                
                # Find the most frequent pair(s)
                max_count = max(pair_count.values())
                candidates = [k for k, v in pair_count.items() if v == max_count]
                best_pair = max(candidates)

                a, b = best_pair

                # Create new token
                new_token = a + b
                self.vocab[next_id] = new_token
                next_id += 1

                # Apply the merge to all pre-tokenized sequences
                # 收集变更
                changes = []
                for token, cnt in pre_tokens_cnt.items():
                    # Find all occurrences of the `best_pair` in `token`
                    indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
                    if indices:
                        # Replace each occurrence with `new_token`
                        new_pre_token = []
                        i = 0
                        while i < len(token):
                            if i in indices:
                                new_pre_token.append(new_token)
                                i += 2
                            else:
                                new_pre_token.append(token[i])
                                i += 1
                        new_pre_token = tuple(new_pre_token)
                        changes.append((token, new_pre_token, cnt))

                # 应用变更
                for old_token, new_pre_token, cnt in changes:
                    pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
                    del pre_tokens_cnt[old_token]

                # Record the merge
                self.merges.append((a, b))

                
                # max_pair = ("", "")
                # max_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
                # # if pair_counts[max_pair] == 0:
                # #     break
                # max_pair_bytes = (max_pair[0].encode("utf-8"), max_pair[1].encode("utf-8"))
                # self.merges.append(max_pair_bytes)


                # if max_pair[0] + max_pair[1] not in self.vocab:
                #     self.vocab[len(self.vocab)] = (max_pair[0] + max_pair[1]).encode("utf-8")
                # 预分词优化
                # update(corpus, max_pair, word_counter, token_map, pair_counts)





    def encode(self, text: str) -> list[int]:
        # return [self.vocab[token] for token in text.split()]

        return None

    def encode_iterable(self, iterable: Iterable[str]) -> list[int]:
        return [self.encode(t) for t in iterable]

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.vocab[token].decode("utf-8") for token in tokens])


# bpe = BPE(vocab_size=800,special_tokens=['<|endoftext|>'])
# bpe.train("/Users/xiaoming.wang/Documents/shawn/课程/CS336/Lab/assignment1-basics/tests/fixtures/tinystories_sample.txt")
# print(bpe.merges)
# print(bpe.vocab)


# bpe = BPE(vocab_size=800,special_tokens=['<|endoftext|>'])




