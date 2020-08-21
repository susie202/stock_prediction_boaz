import numpy as np
import pandas as pd
import os
from torch import optim
import random
import torch
import torch.nn as nn
from behavior_view.NTN.ntn import NeuralTensorNetwork, Load_Data
import argparse
# from Ta_Lib import TaLib

def shuffle(dict1):
    for key in dict1.keys():
        random.seed(42)
        random.shuffle(dict1[key])

def train(epoch, lr, data_loader):
    ntn.train()
    train_loss = 0.0

    for i, train_pos, train_pos_label, train_neg, train_neg_label in data_loader('train'):
        train_pos = train_pos.to(device)
        train_pos_label = train_pos_label.to(device)
        train_neg = train_neg.to(device)
        train_neg_label = train_neg_label.to(device)

        optimizer.zero_grad()

        output_pos = ntn(train_pos)
        output_neg = ntn(train_neg)
        loss = ntn.criterion(output_pos, output_neg, ntn, regularization=lr).to(device)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'epoch : {epoch} / loss : {train_loss/i :,.2f}')

    return train_loss, ntn

def evals(model, data_loader):
    model.eval()
    final = pd.DataFrame([])
    for i, train_pos, train_date in data_loader('eval'):
        train_pos = train_pos.to(device)
        train_pos_date = train_date

        result = model(train_pos)
        
        tmp = pd.concat([pd.DataFrame(result.cpu().detach().numpy()), pd.DataFrame(train_pos_date)], axis=1)

        final = pd.concat([final, tmp])

    return final

def detach(data):
    data = data.cpu().numpy()
    return data

def attach(data):
    data = torch.from_numpy(data).cuda()
    return data

def to_str(data):
    data = str(data)
    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True, type= str,  help= '산업 이름')
    parser.add_argument('--c', required=True, type= str,  help= '기업 이름')
    parser.add_argument('keywordstxt', help= 'keywords txt file')

    args = parser.parse_args()

    with open(args.keywordstxt, encoding='utf-8') as file1:
        keywordstxt = file1.read().splitlines()
    industry = args.i
    company = args.c

    batch_size = 1024
    regularization = 0.00001
    lr = 0.01
    ntn = NeuralTensorNetwork(768, 8)
    device = "cuda"

    svo_pos_vector = pd.DataFrame()
    for keyword in keywordstxt:
        for (root, dirs, files) in os.walk(r'G:\공유 드라이브\Boad ADV Stock\\'):
            for file_name in files:
                if file_name == "svo_pos_{}.pt".format(keyword):
                    print(root + "\\" + file_name)
                    svo_pos_keyword = torch.load(root + "\\" + file_name)
                    svo_pos_keyword = pd.DataFrame(svo_pos_keyword)
                    for col in ['s_pos', 'v_pos', 'o_pos']:
                        svo_pos_keyword[col] = svo_pos_keyword[col].apply(detach)
                    svo_pos_vector = pd.concat([svo_pos_vector, svo_pos_keyword])
                    del svo_pos_keyword
                    

    svo_pos_vector = svo_pos_vector.to_dict('list')

    shuffle(svo_pos_vector)

    ntn.to(device)
    data_loader = Load_Data(svo_pos_vector, batch_size)
    optimizer = optim.Adam(ntn.parameters(), lr=lr)

    epochs= 500
    early_stop = [np.inf, np.inf, np.inf]
    k = 0
    for epoch in range(epochs):
        loss, ntn = train(epoch, lr, data_loader)
        if early_stop[-1] > loss:
            print(early_stop)
            early_stop.pop(0)
            early_stop.append(loss)
            k = 0
            print(early_stop)
            torch.save(ntn, r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_no_{}.pt".format(industry, company))
        k += 1
        if (k>3) & (epoch > 40):
            break

    model = torch.load(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_no_{}.pt".format(industry, company))
    model.cuda()
    final = evals(model, data_loader)
    final.columns = ['event{}'.format(i) for i in range(1, 9)] + ['date']
    real_final = final.groupby('date').mean()

    real_final.to_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_no_{}.xlsx".format(industry, company))
    
    real_final = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_no_{}.xlsx".format(industry, company))
    ta = TaLib(industry, company)
    tech_indi = ta.add_tech_stats().reset_index().rename(columns = {'Frequency' : 'date'}) 
    real_final['date'] = [pd.Timestamp(int(a[:4]), int(a[5:7]), int(a[8:10])) for a in real_final['date']]

    guckje = pd.read_excel(r"G:\공유 드라이브\Boad ADV Stock\국제정세\event_embbed_33_국제정세.xlsx")
    guckje.columns = ['date'] + ['event{}'.format(i) for i in range(9, 17)]
    guckje['date'] = [pd.Timestamp(int(a[:4]), int(a[5:7]), int(a[8:10])) for a in guckje['date']]
    real_real_final = pd.merge(tech_indi, real_final, on=["date"], how='left')
    real_real_final = pd.merge(real_real_final, guckje, on=["date"], how='left')

    real_real_final.to_excel(r"G:\공유 드라이브\Boad ADV Stock\{}\event_embbed_with_tech_no_{}.xlsx".format(industry, company))
