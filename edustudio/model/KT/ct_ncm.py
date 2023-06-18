# -*-coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel



class CT_NCM(GDBaseModel):
    default_cfg = {
        'hidden_size': 64,
        'embed_size': 64,
        'prelen1': 256,  # the first-second layer of performance prediction.
        'prelen2': 128,
        'dropout1': 0,
        'dropout2': 0,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.problem_num = self.datafmt_cfg['dt_info']['exer_count']  # 习题数量
        self.skill_num = self.datafmt_cfg['dt_info']['cpt_count']  # 知识点数量
        self.device = self.trainfmt_cfg['device']
        self.hidden_size = self.model_cfg['hidden_size']
        self.embed_size = self.model_cfg['embed_size']
        self.knowledge_dim = self.hidden_size
        self.input_len = self.knowledge_dim
        self.prelen1 = self.model_cfg['prelen1']
        self.prelen2 = self.model_cfg['prelen2']
        self.loss_function = torch.nn.BCELoss()

    def build_model(self):
        self.dropout1 = nn.Dropout(p=self.model_cfg['dropout1'])
        self.dropout2 = nn.Dropout(p=self.model_cfg['dropout2'])

        self.inter_embedding = torch.nn.Embedding(2 * self.skill_num, self.embed_size)
        self.reclstm = torch.nn.Linear(self.embed_size + self.hidden_size, 7 * self.hidden_size)

        self.problem_disc = torch.nn.Embedding(self.problem_num, 1)
        self.problem_diff = torch.nn.Embedding(self.problem_num, self.knowledge_dim)

        self.linear1 = torch.nn.Linear(self.input_len, self.prelen1)
        self.linear2 = torch.nn.Linear(self.prelen1, self.prelen2)
        self.linear3 = torch.nn.Linear(self.prelen2, 1)
    
    def _init_params(self):
        # super()._init_params()
        pass

    def forward(self, exer_seq, time_lag_seq, cpt_unfold_seq, label_seq, mask_seq, **kwargs):
        problem_seqs_tensor = exer_seq[:,1:].to(self.device)
        skill_seqs_tensor = cpt_unfold_seq.to(self.device)
        time_lag_seqs_tensor = time_lag_seq[:,1:].to(self.device)
        correct_seqs_tensor = label_seq.to(self.device)
        mask_labels = mask_seq.long().to(self.device)
        seqs_length = torch.sum(mask_labels, dim=1)
        delete_row = 0
        for i in range(len(seqs_length)):
            if seqs_length[i] == 1:
                problem_seqs_tensor = problem_seqs_tensor[torch.arange(problem_seqs_tensor.size(0))!=i-delete_row] 
                skill_seqs_tensor = skill_seqs_tensor[torch.arange(skill_seqs_tensor.size(0))!=i-delete_row] 
                time_lag_seqs_tensor = time_lag_seqs_tensor[torch.arange(time_lag_seqs_tensor.size(0))!=i-delete_row] 
                correct_seqs_tensor = correct_seqs_tensor[torch.arange(correct_seqs_tensor.size(0))!=i-delete_row] 
                mask_labels = mask_labels[torch.arange(mask_labels.size(0))!=i-delete_row] 
                delete_row = delete_row + 1

        correct_seqs_tensor = torch.where(mask_labels == 0, -1, correct_seqs_tensor)
        skill_seqs_tensor = torch.where(mask_labels == 0, 0, skill_seqs_tensor)
        mask_labels_temp = mask_labels[:,1:]
        time_lag_seqs_tensor = torch.where(mask_labels_temp == 0, 0, time_lag_seqs_tensor)
        problem_seqs_tensor = torch.where(mask_labels_temp == 0, 0, problem_seqs_tensor)
        # for i in range(mask_labels.shape[0]):
        #     for j in range(mask_labels.shape[1]):
        #         if mask_labels[i][j] == 0:
        #             correct_seqs_tensor[i][j] = -1
        #             skill_seqs_tensor[i][j] = 0
        #             if j>0:
        #                 time_lag_seqs_tensor[i][j-1] = 0
        #                 problem_seqs_tensor[i][j-1] = 0
        seqs_length = torch.sum(mask_labels, dim=1)

        inter_embed_tensor = self.inter_embedding(skill_seqs_tensor + self.skill_num * mask_labels)
        batch_size = correct_seqs_tensor.size()[0]

        hidden, _ = self.continues_lstm(inter_embed_tensor, time_lag_seqs_tensor, seqs_length, batch_size)
        hidden_packed = torch.nn.utils.rnn.pack_padded_sequence(hidden[1:,],
                                                                seqs_length.cpu() - 1,
                                                                batch_first=False,
                                                                enforce_sorted=False)  # 这里有点变动
        theta = hidden_packed.data
        problem_packed = torch.nn.utils.rnn.pack_padded_sequence(problem_seqs_tensor,
                                                                 seqs_length.cpu() - 1,
                                                                 batch_first=True,
                                                                 enforce_sorted=False)
        predictions = torch.squeeze(self.problem_hidden(theta, problem_packed.data))
        labels_packed = torch.nn.utils.rnn.pack_padded_sequence(correct_seqs_tensor[:,1:],
                                                                seqs_length.cpu() - 1,
                                                                batch_first=True,
                                                                enforce_sorted=False)
        labels = labels_packed.data
        #  predictions = torch.where(torch.isnan(predictions), torch.full_like(predictions, 0.5), predictions)
        out_dict = {'predictions': predictions, 'labels': labels}
        return out_dict

    def continues_lstm(self, inter_embed_tensor, time_lag_seqs_tensor, seqs_length, batch_size):
        self.init_states(batch_size=batch_size)
        h_list = []
        h_list.append(self.h_delay)
        for t in range(max(seqs_length) - 1):
            one_batch = inter_embed_tensor[:, t]
            c, self.c_bar, output_t, delay_t = \
                self.conti_lstm(one_batch, self.h_delay, self.c_delay,
                                self.c_bar)
            time_lag_batch = time_lag_seqs_tensor[:, t]
            self.c_delay, self.h_delay = \
                self.delay(c, self.c_bar, output_t, delay_t, time_lag_batch)
            self.h_delay = torch.as_tensor(self.h_delay, dtype=torch.float)
            h_list.append(self.h_delay)
        hidden = torch.stack(h_list)

        return hidden, seqs_length

    def init_states(self, batch_size):
        self.h_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_bar = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)

    def conti_lstm(self, one_batch_inter_embed, h_d_t, c_d_t, c_bar_t):
        input = torch.cat((one_batch_inter_embed, h_d_t), dim=1)
        (i, f, z, o, i_bar, f_bar, delay) = torch.chunk(self.reclstm(input), 7, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        z = torch.tanh(z)
        o = torch.sigmoid(o)
        i_bar = torch.sigmoid(i_bar)
        f_bar = torch.sigmoid(f_bar)
        delay = F.softplus(delay)
        c_t = f * c_d_t + i * z
        c_bar_t = f_bar * c_bar_t + i_bar * z
        return c_t, c_bar_t, o, delay

    def delay(self, c, c_bar, output, delay, time_lag):
        c_delay = c_bar + (c - c_bar) * torch.exp(- delay * time_lag.unsqueeze(-1))
        h_delay = output * torch.tanh(c_delay)
        return c_delay, h_delay

    def problem_hidden(self, theta, problem_data):
        problem_diff = torch.sigmoid(self.problem_diff(problem_data))
        problem_disc = torch.sigmoid(self.problem_disc(problem_data))
        input_x = (theta - problem_diff) * problem_disc * 10
        input_x = self.dropout1(torch.sigmoid(self.linear1(input_x)))
        input_x = self.dropout2(torch.sigmoid(self.linear2(input_x)))
        output = torch.sigmoid(self.linear3(input_x))
        return output

    def predict(self, **kwargs):
        outdict = self(**kwargs)
        return {
            'y_pd': outdict['predictions'],
            'y_gt': torch.as_tensor(outdict['labels'], dtype=torch.float)
        }

    def get_main_loss(self, **kwargs):
        outdict = self(**kwargs)
        predictions = outdict['predictions']
        labels = outdict['labels']
        labels = torch.as_tensor(labels, dtype=torch.float)
        loss = self.loss_function(predictions, labels)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
