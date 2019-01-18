#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from text_utils import TextDataset, textDataLoader
import logging as logger
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="train_lstm.log",filemode="w")

torch.manual_seed(1)




# I use glove to obtain word embedding representation
glove_file="glove.6B.300d.txt"
train_data = TextDataset("ag_news_csv/train.csv", glove_file, label_shift=1)
train_loader = textDataLoader(train_data, batch_size=128*4, shuffle=True, num_workers=4)
test_data = TextDataset("ag_news_csv/test.csv", glove_file, label_shift=1)
test_loader = textDataLoader(test_data, batch_size=128*4, shuffle=False)
print("len(train_data): %d len(test_data): %d" %(len(train_data), len(test_data)))
weights_matrix = train_data.get_weights_matrix()
EMBEDDING_DIM = weights_matrix.shape[1]
HIDDEN_DIM = 512

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, weights_matrix, tagset_size, num_class=4):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        vocab_size, embedding_dim = weights_matrix.shape
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.load_state_dict({"weight": weights_matrix})
        self.word_embeddings.weight.requires_grad = False

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.num_layers = 1
        self.bidirectional = True
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)

        # The linear layer that maps from hidden state space to tag space
#        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.classifier = nn.Linear(hidden_dim, num_class)

        self.hidden = None #self.init_hidden()

    def init_hidden(self, batch_size, device="cpu"):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (minibatch_size, num_layers, hidden_dim)
        if self.bidirectional:
            return (torch.randn(self.num_layers*2, batch_size, self.hidden_dim, device=device),\
                torch.randn(self.num_layers*2, batch_size, self.hidden_dim, device=device))
        else:
            return (torch.randn(self.num_layers, batch_size, self.hidden_dim, device=device),\
                torch.randn(self.num_layers, batch_size, self.hidden_dim, device=device))


    def forward(self, sentences, lengths):
        batch_size = sentences.shape[0]
        self.hidden = self.init_hidden(batch_size, device=sentences.device)
        embeddings = self.word_embeddings(sentences)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        lstm_out, (hidden, cell) = self.lstm(packed, self.hidden)
#        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
#        tag_space = self.hidden2tag(self.hidden[0][-1].view(sentences.shape[0], -1))
#        output = self.classifier(tag_space)

        output = self.classifier(hidden[-1].view(batch_size, -1))

        return output



model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, weights_matrix, 128)
model = nn.DataParallel(model).cuda()
print("LSTM model:", model)

loss_function = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.1)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150],\
            gamma=0.1)

def train(data_loader, epochs=300, print_freq=50):
    model.train()
    for epoch in range(epochs):
        lr_scheduler.step()
        for step, (sentences, lengths, label) in enumerate(data_loader):
            label = label.squeeze()
            label = label.cuda()
            sentences = sentences.cuda()
            lengths = lengths.cuda()
            batch_size = label.shape[0]
            tag_scores = model.forward(sentences, lengths)
            loss = loss_function(tag_scores, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % print_freq == 0:
                logger.info("batch: %d, iter: %d, loss: %.6f" %(epoch, step,  loss.item()))
        test(test_loader, epoch)
        model.train()

def test(data_loader, epoch):
    acc = 0.0
    correct = 0
    model.eval()
    with torch.no_grad():
        for step, (sentences, lengths, label) in enumerate(data_loader):
            label = label.squeeze()
            sentences, lengths, label = sentences.cuda(), lengths.cuda(), label.cuda()
            output = model.forward(sentences, lengths)
            loss = loss_function(output, label)

            pred = output.max(dim=1, keepdim=True)[1]

            correct += pred.eq(label.view_as(pred)).sum().item()

    acc = correct * 1.0 / len(data_loader.dataset)
    logger.info("Epoch:%d test_acc: %.6f" % (epoch, acc))

if __name__ == "__main__":
    train(train_loader, 300)

