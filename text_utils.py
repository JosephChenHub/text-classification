import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
from numpy.random import randint
import nltk
from nltk.tokenize import word_tokenize




class TextDataset(Dataset):
    def __init__(self, file_name, glove_file, label_shift=0):
        self.lines = [line.strip().split("\",\"") for line in open(file_name, "r")]
        self.label_shift = label_shift
        self.__words, self.__weights_matrix = self.__get_word_vectors(glove_file)

        self.__nltk_words = set(nltk.corpus.words.words())

    def __get_word_vectors(self, file_name):
        lines = [line.strip() for line in open(file_name, "r")]
        words = dict()
        vectors = dict()
        dim = 0
        for i, line in enumerate(lines):
            tmp = line.split(" ")
            word = tmp[0]
            vec  = np.asarray([float(x) for x in tmp[1:] ])
            words[word] = i
            vectors[word] = vec
            dim = len(vec)

        weights_matrix = torch.zeros(len(words), dim)
        for i, word in enumerate(words.keys()):
            vec = vectors[word]
            idx = words[word]
            weights_matrix[idx] = torch.from_numpy(vec)

        return words, weights_matrix


    def get_weights_matrix(self):
        return self.__weights_matrix


    def __prepare_sequence(self, seq):
        idxs = [self.__words[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    def __len__(self):
        return len(self.lines)

    def __line_segments(self, line):
        line = line.lower()
        line = line.replace("-", " ")
        words = word_tokenize(line)
        words = [word for word in words if word in self.__words ]

        return words

    def __getitem__(self, index):
        if index > len(self.lines):
            raise IndexError("index exceeds the max-length of the dataset.")
        assert len(self.lines[index]) == 3
        label = int(self.lines[index][0][1:]) - self.label_shift #age csv start from 1
        label_tensor = torch.zeros(1, dtype=torch.long)
        label_tensor[0] = label

        title = self.__line_segments(self.lines[index][1])
        content = self.__line_segments(self.lines[index][2][:-1])

        title_tensor = self.__prepare_sequence(title)
        content_tensor = self.__prepare_sequence(content)
        text_tensor = torch.cat((title_tensor, content_tensor))

        return text_tensor, label_tensor


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths)
        padded_seqs = torch.zeros(len(sequences), max_len, dtype=sequences[0].dtype)
        for i , seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
        return padded_seqs, torch.tensor(lengths, dtype=torch.long)

    data.sort(key=lambda x: len(x[0]), reverse=True)
    text, label = zip(*data)
    label = torch.stack(label, 0)

    text_padded, text_length = merge(text)

    return text_padded, text_length, label


def textDataLoader(dataset, batch_size, shuffle=False, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, collate_fn=collate_fn, num_workers=num_workers)



if __name__ == "__main__":
    from torch.nn.utils.rnn import pack_padded_sequence
    text_data = TextDataset("ag_news_csv/train.csv", "glove.6B.50d.txt", 1)
    data_loader = textDataLoader(text_data, batch_size=128, shuffle=False)
    W = text_data.get_weights_matrix()
    print("W.shape:", W.shape)
    for i, (text_padded, text_length, label) in enumerate(data_loader):
        text, text_batch_size = pack_padded_sequence(text_padded, text_length, batch_first=True)
        if i < 5:
            print("text.shape:", text.shape, "|padded:", text_padded.shape, "|label:", label.shape)
            import pdb;pdb.set_trace()
