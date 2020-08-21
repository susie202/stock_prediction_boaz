import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertModel as BertModel2
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

class BFT_DATA_LOADER(Dataset):
    def __init__(self, maxlen):

        #Store the contents of the file in a pandas dataframe
        # self.df = self.mklabel()
        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'header']
        label = self.df.loc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

    def mklabel(self):
        os.chdir(r"G:\공유 드라이브\Boad ADV Stock")
        data = pd.DataFrame()
        print("mk_label_processing..")
        for indus in ['\\2차전지', '\\축산업', '\\화장품']:
            for f in os.listdir(os.getcwd()+ indus + '\\crawled_data\\'):
                if 'translate' in f:
                    translate_data = pd.read_excel(os.getcwd()+ indus + '\\crawled_data\\' + f)
                    translate_data.date = pd.to_datetime(translate_data.date)
                    if f[:-15] in [ff[:-19] for ff in os.listdir(os.getcwd() + indus + "\\tech_data\\") if "withlabel" in ff]:
                        label = pd.read_csv(os.getcwd() + indus + "\\tech_data\\" + f[:-15] + "_tech_withlabel.csv")
                        label.Symbol = pd.to_datetime(label.Symbol)
                        label.rename(columns = {'Symbol' : 'date'}, inplace = True)
                        tmpdata = pd.merge(translate_data, label, on = 'date')
                        tmpdata = tmpdata[['header', 'date', 'label']]
                        tmpdata.to_excel(os.getcwd() + indus + "\\crawled_data\\" + f[:-15] + "_bft_label.xlsx")
                        data = pd.concat([data, tmpdata], axis=0)
                    else:
                        tmplabel = pd.DataFrame(pd.date_range('2009.01.01', '2020.06.01'), columns=['Symbol'])
                        for withlabel in [ff for ff in os.listdir(os.getcwd() + indus + "\\tech_data\\") if "withlabel" in ff]:
                            ttmplabel = pd.read_csv(os.getcwd() + indus + "\\tech_data\\" + withlabel)[['Symbol','label']]
                            ttmplabel.Symbol = pd.to_datetime(ttmplabel.Symbol)
                            tmplabel = pd.merge(tmplabel, ttmplabel, on='Symbol', how='left')
                        tmplabel.columns = ["date", '1', '2', '3', '4', '5']
                        tmplabel['label'] = tmplabel.median(axis=1, numeric_only=True)
                        tmplabel['label'] = tmplabel['label'].apply(lambda x:0 if x == 0.5 else x)
                        tmpdata = pd.merge(translate_data, tmplabel[['date', 'label']], on = 'date')
                        tmpdata = tmpdata[['header', 'date', 'label']]
                        tmpdata.to_excel(os.getcwd() + indus + "\\crawled_data\\" + f[:-15] + "_bft_label.xlsx")
                        data = pd.concat([data, tmpdata], axis=0)

        data = data.sort_values(by='date')
        data = data.dropna(axis=0)
        data = data.reset_index()

        return data

    # def mklabel_fine_tune(self):
    #     os.chdir(r"G:\공유 드라이브\Boad ADV Stock\bert-base-finetuned")
    #     data = pd.DataFrame()
    #     print("mk_label_processing_for_last_8_days...")
    #     for f in os.listdir(os.getcwd()+"\\fine_tuning\\"):
    #         translate_data = pd.read_excel(os.getcwd() +"\\fine_tuning\\" + f)
    #         translate_data.date = pd.to_datetime(translate_data.date)



class UpDownClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(UpDownClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel2.from_pretrained('bert-base-uncased').cuda()
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1).cuda()

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits

class BertFineTuning:
    def __init__(self):
        self.device = torch.device('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(r'G:\공유 드라이브\Boad ADV Stock\bert-base-finetuned')
        self.model.cuda()
        self.train_set = BFT_DATA_LOADER(150)
        self.train_loader = DataLoader(self.train_set, batch_size = 256)
        self.net = UpDownClassifier(freeze_bert=True)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr = 2e-5)    

    def word_embedding(self, sentence):
    
        tokenized_text = self.tokenizer.tokenize(sentence)
        indexed_token = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_id = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_token]).to(self.device)
        segments_tensor = torch.tensor([segments_id]).to(self.device)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensor)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []
        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor

            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        token_vecs = encoded_layers[11][0]  ## 11 : last layer

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding

    def train(self, epoch = 50):

        for ep in range(epoch):
            for it, (seq, attn_masks, labels) in enumerate(self.train_loader):
                #Clear gradients
                self.optimizer.zero_grad()  
                #Converting these to cuda tensors
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

                #Obtaining the logits from the model
                logits = self.net(seq, attn_masks)

                #Computing loss
                loss = self.criterion(logits.squeeze(-1), labels.float())

                #Backpropagating the gradients
                loss.backward()

                #Optimization step
                self.optimizer.step()

                if (it-1) % 1000 == 0:
                    print("epoch : {}, Iteration : {}, Loss : {}".format(ep, it-1, loss.item()))
        
            self.net.bert_layer.save_pretrained(r'G:\공유 드라이브\Boad ADV Stock\bert-base-finetuned')

if __name__ == "__main__":
    bft = BertFineTuning()
    bft.train()
