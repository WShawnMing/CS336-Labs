import os
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from collections import Counter
from typing import List , Tuple , Iterable


class BPE:
    def __init__(self, vocab_size: int, special_tokens: List[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def train(self, input_path: str | os.PathLike) -> None:
        self.merges = []
        self.vocab = {}
        for token in self.special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")



        byte_to_unicode_map = gpt2_bytes_to_unicode()
        unicode_to_byte_map = {v: k for k, v in byte_to_unicode_map.items()}
        sorted_unicode_chars = sorted(unicode_to_byte_map.keys())

        for unicode_char in sorted_unicode_chars:
            original_byte = unicode_to_byte_map[unicode_char]
            self.vocab[len(self.vocab)] =bytes([original_byte])

        def update(corpus, best_pair,word_counter,word_map,pair_counts):
            best_str = best_pair[0] + best_pair[1]
            for word in word_counter: 
                if word in self.special_tokens:
                    continue
                if best_str not in word:
                    continue
                cor = corpus[word_map[word]]

                new_cor = []
                last_flag = True
                i = 0
                while i < len(cor) - 1:
                    if cor[i] == best_pair[0] and cor[i+1] == best_pair[1]:
                        if i > 0:
                            pre_token = cor[i-1]
                            pair = (pre_token, best_pair[0])
                            pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                            pair = (pre_token, best_pair[0] + best_pair[1])
                            pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]

                        if i+2 < len(cor):
                            next_token = cor[i+2]
                            pair = (best_pair[1], next_token)
                            pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                            pair = (best_pair[0] + best_pair[1], next_token)
                            pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]
    
                        pair = (best_pair[0], best_pair[1])
                        pair_counts[pair] = pair_counts.get(pair, 0) - word_counter[word]
                        new_cor.append(best_str)
                        if i+1 == len(cor) - 1:
                            last_flag = False
                        i += 2
                    else:
                        new_cor.append(cor[i])
                        i += 1
                if last_flag:
                    new_cor.append(cor[-1])
                corpus[word_map[word]] = new_cor
                            
            pair_counts = {k: v for k, v in pair_counts.items() if v > 0}


        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, 1, "<|endoftext|>".encode("utf-8"))
            boundary_start, boundary_end = boundaries[0], boundaries[1]
            chunk = f.read(boundary_end - boundary_start).decode("utf-8", errors="ignore")

            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            special_tokens_pattern = r'|'.join(escaped_tokens)

            word_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            full_pattern = f"{special_tokens_pattern}|{word_pattern}"
            words = [match.group(0) for match in re.finditer(full_pattern, chunk)]





            word_counter = Counter(words)

            corpus = []
            token_map = {}

    
            for word in word_counter:
                if word in self.special_tokens:
                    word_counter[word] = 0
                    continue
                path =[]
                for i in range(len(word)):
                    path.append(word[i])
                corpus.append(path)
                token_map[word] = len(corpus) - 1

            # print(corpus)
            # print(token_map)

            pair_counts = {}

            # 构建paircounts
            for word in word_counter:
                if word in self.special_tokens:
                    continue
                word_path = corpus[token_map[word]]
                for i in range(len(word_path) - 1):
                    pair = (word_path[i], word_path[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]

            while len(self.vocab) < self.vocab_size :
                max_pair = ("", "")

                # if len(pair_counts) == 0:
                #     for word in word_counter:
                #         word_path = corpus[token_map[word]]
                #         for i in range(len(word_path) - 1):
                #                 pair = (word_path[i], word_path[i+1])
                #                 pair_counts[pair] = pair_counts.get(pair, 0) + word_counter[word]
                #                 if pair_counts[pair] > max_count:
                #                     max_count = pair_counts[pair]
                #                     max_pair = pair
                #                 # 字典序
                #                 if pair_counts[pair] == max_count :
                #                     max_pair = pair if max_pair > pair else max_pair
                #     # print(pair_counts)
                # else:
                # for pair in pair_counts:
                #     if pair_counts[pair] > max_count:
                #         max_count = pair_counts[pair]
                #         max_pair = pair
                max_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
                if pair_counts[max_pair] == 0:
                    break
                max_pair_bytes = (max_pair[0].encode("utf-8"), max_pair[1].encode("utf-8"))
                self.merges.append(max_pair_bytes)
                if max_pair[0] + max_pair[1] not in self.vocab:
                    self.vocab[len(self.vocab)] = (max_pair[0] + max_pair[1]).encode("utf-8")
                
                update(corpus, max_pair, word_counter, token_map, pair_counts)


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