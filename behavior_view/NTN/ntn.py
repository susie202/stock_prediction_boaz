import torch
import torch.nn as nn
import numpy as np
import random, copy

class NeuralTensorNetwork(nn.Module):

    def __init__(self, embedding_size, tensor_dim, device="cuda"):
        super(NeuralTensorNetwork, self).__init__()

        self.device = device
        self.tensor_dim = tensor_dim

        ##Tensor Weight
        # |T1| = (embedding_size, embedding_size, tensor_dim)
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)

        # |T2| = (embedding_size, embedding_size, tensor_dim)
        self.T2 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T2.data.normal_(mean=0.0, std=0.02)

        # |T3| = (tensor_dim, tensor_dim, tensor_dim)
        self.T3 = nn.Parameter(torch.Tensor(tensor_dim * tensor_dim * tensor_dim))
        self.T3.data.normal_(mean=0.0, std=0.02)

        # |W1| = (embedding_size * 2, tensor_dim)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W2| = (embedding_size * 2, tensor_dim)
        self.W2 = nn.Linear(embedding_size * 2, tensor_dim)

        # |W3| = (tensor_dim * 2, tensor_dim)
        self.W3 = nn.Linear(tensor_dim * 2, tensor_dim)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0)


    def forward(self, svo):

        subj = svo[0]
        verb = svo[1]
        obj = svo[2]

        R1 = self.tensor_Linear(subj, verb, self.T1, self.W1)
        R1 = self.tanh(R1)
        R1 = self.dropout(R1)
        # |R1| = (batch_size, tensor_dim)

        R2 = self.tensor_Linear(verb, obj, self.T2, self.W2)
        R2 = self.tanh(R2)
        R2 = self.dropout(R2)
        # |R2| = (batch_size, tensor_dim)

        U = self.tensor_Linear(R1, R2, self.T3, self.W3)
        U = self.tanh(U)

        return U


    def tensor_Linear(self, o1, o2, tensor_layer, linear_layer):
        # |o1| = (batch_size, unknown_dim)
        # |o2| = (batch_size, unknown_dim)
        # |tensor_layer| = (unknown_dim * unknown_dim * tensor_dim)
        # |linear_layer| = (unknown_dim * 2, tensor_dim)

        batch_size, unknown_dim = o1.size()  # ( unknown_dim : 768 - word_vecter_dim )

        # 1. Linear Production
        o1_o2 = torch.cat((o1, o2), dim=1)
        # |o1_o2| = (batch_size, unknown_dim * 2)
        linear_product = linear_layer(o1_o2)
        # |linear_product| = (batch_size, tensor_dim)

        # 2. Tensor Production ( mm : 행렬과 행렬의 곱 )
        tensor_product = o1.mm(tensor_layer.view(unknown_dim, -1))
        # |tensor_product| = (batch_size, unknown_dim * tensor_dim)
        tensor_product = tensor_product.view(batch_size, -1, unknown_dim).bmm(o2.unsqueeze(1).permute(0,2,1).contiguous()).squeeze()
        tensor_product = tensor_product.contiguous()
        # |tensor_product| = (batch_size, tensor_dim)

        # 3. Summation
        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)

        return result

    def criterion(self, output_pos, output_neg, model, regularization=0.0001, device = 'cuda'):

        ## element 별로 들어감
        loss1 = torch.max(torch.sub(output_neg, output_pos) + 1, torch.tensor(0.0).to(device))
        loss1 = torch.sum(loss1)
        loss2 = torch.sqrt(np.sum(np.array([torch.sum(torch.square(var)) for var in model.parameters()])))
        loss = loss1 + (regularization * loss2)

        return loss

class Load_Data:
    def __init__(self, svo_pos, batch_size):
        self.svo_neg = copy.deepcopy(svo_pos)
        self.svo_neg['s_neg'] = self.svo_neg.pop('s_pos')
        self.svo_neg['v_neg'] = self.svo_neg.pop('v_pos')
        self.svo_neg['o_neg'] = self.svo_neg.pop('o_pos')
        self.svo_neg['label'] = self.svo_neg.pop('label')
        self.svo_neg['date'] = self.svo_neg.pop('date')
        self.svo_neg['s_neg'] = []
        for i in range(svo_pos['s_pos'].__len__()):
            self.svo_neg['s_neg'].append(random.choice(svo_pos['s_pos']))
        self.data_len = int(len(svo_pos['s_pos']))
        self.batch_size = batch_size
        self.train_pos_all = np.array(list(svo_pos.values()))[:3, :]
        self.train_label_pos_all = np.array(list(svo_pos.values()))[3:,:]
        self.train_date = np.array(list(svo_pos.values()))[4:,:]
        self.train_neg_all = np.array(list(self.svo_neg.values()))[:3, :]
        self.train_label_neg_all = np.array(list(svo_pos.values()))[3:,:]

        # self.test_pos_all = np.array(list(svo_pos.values()))[:3, int(len(svo_pos['s_pos']) * 0.8) :]
        # self.test_label_pos_all = np.array(list(svo_pos.values()))[3:, int(len(svo_pos['s_pos']) * 0.8) :]
        # self.test_neg_all = np.array(list(svo_neg.values()))[:3, int(len(svo_neg['s_neg']) * 0.8) :]
        # self.test_label_neg_all = np.array(list(svo_neg.values()))[3:, int(len(svo_neg['s_neg']) * 0.8) :]

    def __call__(self, types):
        if types == "train":
            i = 0
            while i < int(self.data_len/self.batch_size):
                train_pos = self.train_pos_all[:, i*self.batch_size : (i+1)*self.batch_size]
                train_label_pos = np.array(list(self.train_label_pos_all[0, :]), dtype=np.float)[i*self.batch_size:(i+1)*self.batch_size]
                train_neg = self.train_neg_all[:, i*self.batch_size : (i+1)*self.batch_size]
                train_label_neg = -np.array(list(self.train_label_neg_all[0, :]), dtype=np.float)[i*self.batch_size:(i+1)*self.batch_size]
                i+= 1
                train_pos = [[data for data in t_pos] for t_pos in train_pos]
                train_neg = [[data for data in t_pos] for t_pos in train_neg]
                train_pos = np.array(train_pos)
                train_neg = np.array(train_neg)
                yield i, torch.from_numpy(train_pos), torch.from_numpy(train_label_pos), torch.from_numpy(train_neg), torch.from_numpy(train_label_neg)

        elif types == "eval":
            i = 0
            while i < int(self.data_len/self.batch_size):
                train_pos = self.train_pos_all[:, i*self.batch_size : (i+1)*self.batch_size]
                train_date = np.array(list(self.train_date[0, :])[i*self.batch_size:(i+1)*self.batch_size])
                i+= 1
                train_pos = [[data for data in t_pos] for t_pos in train_pos]
                train_pos = np.array(train_pos)
                yield i, torch.from_numpy(train_pos), train_date

