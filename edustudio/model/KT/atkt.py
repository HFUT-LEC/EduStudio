from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class ATKT(GDBaseModel):
    default_cfg = {
        'skill_dim': 256,
        'answer_dim': 96,
        'hidden_dim': 80,
        'attention_dim': 80,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def _init_params(self):
        # super()._init_params()
        pass

    def build_cfg(self):
        self.output_dim = self.datafmt_cfg['dt_info']['cpt_count']
        self.skill_dim = self.model_cfg['skill_dim']
        self.answer_dim = self.model_cfg['answer_dim']
        self.hidden_dim = self.model_cfg['hidden_dim']
        self.attention_dim = self.model_cfg['attention_dim']

    def build_model(self):
        self.rnn = nn.LSTM(self.skill_dim+self.answer_dim, self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.sig = nn.Sigmoid()
        
        self.skill_emb = nn.Embedding(self.output_dim+1, self.skill_dim)
        self.skill_emb.weight.data[-1]= 0
        
        self.answer_emb = nn.Embedding(2+1, self.answer_dim)
        self.answer_emb.weight.data[-1]= 0

        self.mlp = nn.Linear(self.hidden_dim, self.attention_dim)
        self.similarity = nn.Linear(self.attention_dim, 1, bias=False)

    def _get_next_pred(self, res, skill):
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        att_w = self.mlp(lstm_output)
        att_w = torch.tanh(att_w)
        att_w = self.similarity(att_w)
        
        alphas=nn.Softmax(dim=1)(att_w)
        
        attn_ouput=alphas*lstm_output
        attn_output_cum=torch.cumsum(attn_ouput, dim=1)
        attn_output_cum_1=attn_output_cum-attn_ouput

        final_output=torch.cat((attn_output_cum_1, lstm_output),2)
        
        return final_output

    def forward(self, cpt_unfold_seq, label_seq, mask_seq, p_adv=None, **kwargs):
        skill = torch.where(mask_seq==0, torch.tensor([self.output_dim]), cpt_unfold_seq)
        answer = torch.where(mask_seq==0, torch.tensor([2]), label_seq).long()
        perturbation = p_adv
        
        skill_embedding=self.skill_emb(skill)  # skill:24*500(batch_size*seq_len), 若知识点所对应的序列学生没有做，则转换为self.output_dim
        answer_embedding=self.answer_emb(answer)  # answer:24*500, 若学生没有做对应的习题，则转化为2
        
        skill_answer=torch.cat((skill_embedding,answer_embedding), 2)
        answer_skill=torch.cat((answer_embedding,skill_embedding), 2)
        
        answer=answer.unsqueeze(2).expand_as(skill_answer)
        
        skill_answer_embedding=torch.where(answer==1, skill_answer, answer_skill)

        skill_answer_embedding1=skill_answer_embedding
        
        if  perturbation is not None:
            skill_answer_embedding+=perturbation
        
        out,_ = self.rnn(skill_answer_embedding)
        out=self.attention_module(out)
        res = self.sig(self.fc(out))

        res = res[:, :-1, :]
        pred_res = self._get_next_pred(res, skill)  # 24*499(batch_size*(seq_len-1))
        
        return pred_res, skill_answer_embedding1

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd, _ = self(**kwargs)
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = None
        if kwargs.get('label_seq', None) is not None:
            y_gt = kwargs['label_seq'][:, 1:]
            y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        return {
            'y_pd': y_pd,
            'y_gt': y_gt
        }



