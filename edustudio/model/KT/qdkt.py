r"""
QDKT
##########################################

Reference:
    Shashank Sonkar et al. "qDKT: Question-Centric Deep Knowledge Tracing" in EDM 2020.

Reference Code:
    https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/qdkt.py

"""
from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class QDKT(GDBaseModel):
    r"""
   QDKT

   default_cfg:
      'emb_size': 100  # dimension of embedding
      'dropout_rate': 0.2      # dropout rate
      'hidden_size': 100        # hidden size of LSTM
        'num_layers': 1         # num layers of LSTM
        'w_reg':0.01            # weight of loss_reg
   """
    default_cfg = {
        'emb_size': 100,
        'hidden_size': 100,
        'num_layers': 1,
        'dropout_rate': 0.2,
        'w_reg':0.01
    }

    def add_extra_data(self, **kwargs):
        self.laplacian_matrix = kwargs['laplacian_matrix']
        self.laplacian = torch.tensor(self.laplacian_matrix).to(self.device).float()
        self.train_dict = kwargs['train_dict']

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.w_reg = self.modeltpl_cfg['w_reg']

    def _init_params(self):
        super()._init_params()
        for w in self.fasttext_model.words:
            label = int(w[-1])
            index = int(w[:-1])
            fast_weight = self.fasttext_model[w]
            new_index = index+label*index
            sd = self.exer_emb.state_dict()
            sd['weight'][new_index].copy_(torch.tensor(fast_weight))

    def build_model(self):
        self.exer_emb = nn.Embedding(
            self.n_item * 2, self.modeltpl_cfg['emb_size']
        )
        self.seq_model = nn.LSTM(
            self.modeltpl_cfg['emb_size'], self.modeltpl_cfg['hidden_size'],
            self.modeltpl_cfg['num_layers'], batch_first=True
        )

        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['dropout_rate'])
        self.fc_layer = nn.Linear(self.modeltpl_cfg['hidden_size'], self.n_item)

        f = open("tmp.txt", "w")
        for l in self.train_fasttext():
            f.writelines(l)
        f.close()
        import fasttext
        self.fasttext_model = fasttext.train_unsupervised("tmp.txt",  dim = self.modeltpl_cfg['emb_size'],minCount=1, wordNgrams=2)

    def train_fasttext(self):
        stu_ids = self.train_dict['stu_id']
        exer_ids = self.train_dict['exer_seq']
        label_ids = self.train_dict['label_seq']
        mask_ids = self.train_dict['mask_seq']
        len_s = len(stu_ids)
        ex_stuid = -1
        all_str = []
        for i in range(len_s):
            tmp_stu_id = stu_ids[i]
            if ex_stuid != tmp_stu_id:
                ex_stuid = tmp_stu_id
                if i !=0:
                    all_str.append(s_str)
                s_str = []
            tmp_exer = exer_ids[i, :]
            tmp_label = label_ids[i, :]
            tmp_mask = mask_ids[i, :]
            for mk in range(len(tmp_mask)):
                if tmp_mask[mk] != 0:  # 有效
                    tmp_str = str(tmp_exer[mk]) + str(int(tmp_label[mk]))+" "
                    s_str.append(tmp_str)
        return all_str

    def forward(self, exer_seq, label_seq, **kwargs):
        input_x = self.exer_emb(exer_seq + label_seq.long() * self.n_item)
        output, _ = self.seq_model(input_x)
        output = self.dropout_layer(output)
        y_pd = self.fc_layer(output).sigmoid()
        return y_pd
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }

    def get_main_loss(self, **kwargs):
        y_pd = self(**kwargs)
        y1 = y_pd[:, -1, :].unsqueeze(1)
        y2 = y_pd[:, -1, :].unsqueeze(2)
        ls1 = torch.matmul(y1, self.laplacian)
        ls2 = torch.bmm(ls1, y2).squeeze(-1).squeeze(-1)
        loss_reg = torch.mean(ls2)
        y_pd = y_pd[:, :-1].gather(
            index=kwargs['exer_seq'][:, 1:].unsqueeze(dim=-1), dim=2
        ).squeeze(dim=-1)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_reg':loss_reg*self.w_reg,
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
