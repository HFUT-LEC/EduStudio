from edustudio.model import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class MF(GDBaseModel):
    default_cfg = {
        'emb_dim': 32,
        'reg_user': 0.0,
        'reg_item': 0.0
    }

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.emb_size = self.modeltpl_cfg['emb_dim']
        self.reg_user = self.modeltpl_cfg['reg_user']
        self.reg_item = self.modeltpl_cfg['reg_item']

    def build_model(self):
        self.user_emb = nn.Embedding(
            num_embeddings=self.n_user,
            embedding_dim=self.emb_size
        )
        self.item_emb = nn.Embedding(
            num_embeddings=self.n_item,
            embedding_dim=self.emb_size
        )

    def forward(self, user_idx: torch.LongTensor, item_idx: torch.LongTensor):
        assert len(user_idx.shape) == 1 and len(item_idx.shape) == 1 and user_idx.shape[0] == item_idx.shape[0]
        return torch.einsum("ij,ij->i", self.user_emb(user_idx), self.item_emb(item_idx)).sigmoid()

    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        return {
            'y_pd': self(stu_id, exer_id).flatten(),
        }

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        pd = self(stu_id, exer_id).flatten()
        loss = F.binary_cross_entropy(input=pd, target=label)
        return {
            'loss_main': loss
        }
        
    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)