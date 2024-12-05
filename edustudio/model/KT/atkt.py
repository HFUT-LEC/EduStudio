r"""
ATKT
##################################
Reference:
    Xiaopeng Guo et al. "Enhancing knowledge tracing via adversarial training." in MM 2021.
Reference code:
    https://github.com/xiaopengguo/ATKT
"""

from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class ATKT(GDBaseModel):
    """
    skill_dim: dimensions of knowledge concept representation
    answer_dim: representation dimensions of students' answers to exercises
    hidden_dim: dimensions of the hidden layer between the convolutional neural network and the output layer
    attention_dim: dimensions of parameters in the attention mechanism
    """
    default_cfg = {
        'skill_dim': 256,
        'answer_dim': 96,
        'hidden_dim': 80,
        'attention_dim': 80,
    }

    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def _init_params(self):
        """Skip initialization of parameters for individual components of the model"""
        pass

    def build_cfg(self):
        """Initialize the parameters of the model"""
        self.output_dim = self.datatpl_cfg['dt_info']['cpt_count']
        self.skill_dim = self.modeltpl_cfg['skill_dim']
        self.answer_dim = self.modeltpl_cfg['answer_dim']
        self.hidden_dim = self.modeltpl_cfg['hidden_dim']
        self.attention_dim = self.modeltpl_cfg['attention_dim']

    def build_model(self):
        """Initialize the various components of the model"""
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
        """Get how well the model predicts students' responses to exercise questions

        Args:
            res (torch.Tensor): Shape of [batch_size, seq_len-1, 123]
            skill (torch.Tensor): Sequence of knowledge concepts related to exercises. Shape of [batch_size, seq_len]

        Returns:
            torch.Tensor: the model predictions of students' responses to exercise questions. Shape of [batch_size, seq_len-1]
        """
        one_hot = torch.eye(self.output_dim, device=res.device)
        one_hot = torch.cat((one_hot, torch.zeros(1, self.output_dim).to(self.device)), dim=0)
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, one_hot)
        
        pred = (res * one_hot_skill).sum(dim=-1)
        return pred
    
    def attention_module(self, lstm_output):
        """

        Args:
            lstm_output (torch.Tensor): output of lstm. Shape of [batch_size, seq_len, attention_dim]

        Returns:
            torch.Tensor: output of attention module. Shape of [batch_size, seq_len, 2*attention_dim]
        """
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
        """A function of how well the model predicts students' responses to exercise questions

        Args:
            cpt_unfold_seq (torch.Tensor): Sequence of knowledge concepts related to exercises. Shape of [batch_size, seq_len]
            label_seq (torch.Tensor): Sequence of students' answers to exercises. Shape of [batch_size, seq_len]
            mask_seq (torch.Tensor): Sequence of mask. Mask=1 indicates that the student has answered the exercise, otherwise vice versa. Shape of [batch_size, seq_len] 
            p_adv (torch.Tensor, optional): perturbation to skill_answer_embedding. Defaults to None.

        Returns:
            torch.Tensor: The predictions of the model and the skill answer embedding
        """
        skill = torch.where(mask_seq==0, torch.tensor([self.output_dim]).to(self.device), cpt_unfold_seq)
        answer = torch.where(mask_seq==0, torch.tensor([2]).to(self.device), label_seq).long()
        perturbation = p_adv
        
        skill_embedding=self.skill_emb(skill)  # skill:24*500(batch_size*seq_len)
        answer_embedding=self.answer_emb(answer)  # answer:24*500
        
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
        """A function of get how well the model predicts students' responses to exercise questions and the groundtruth

        Returns:
            dict: The predictions of the model and the real situation
        """
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



