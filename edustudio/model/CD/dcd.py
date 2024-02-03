import torch
import torch.nn as nn
import torch.nn.functional as F
from edustudio.model import GDBaseModel
from edustudio.model.utils.components import MLP, PosMLP
import math
import math
from torch.autograd import Function


class MarginLossZeroOne(nn.Module):
    def __init__(self, margin=0.5, reduction: str = 'mean') -> None:
        assert reduction in ['mean', 'sum', 'none']
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, pos_pd, neg_pd):
        logits = self.margin - (pos_pd - neg_pd)
        logits[logits < 0] = 0.0
        if self.reduction == 'mean':
            return logits.mean()
        elif self.reduction == 'sum':
            return logits.sum()
        else:
            return logits


class DCD(GDBaseModel):
    default_cfg = {
        'EncoderUserHidden': [512],
        'EncoderItemHidden': [512],
        'lambda_main': 1.0,
        'lambda_q': 1.0,
        'align_margin_loss_kwargs': {'margin': 0.7, 'topk': 2, "d1":1, 'margin_lambda': 0.5, 'norm': 1, 'norm_lambda': 1.0, 'start_epoch': 1},
        'sampling_type': 'mws',
        'b_sample_type': 'gumbel_softmax',
        'b_sample_kwargs': {'tau': 1.0, 'hard': True},
        'bernoulli_prior_p': 0.1,
        'bernoulli_prior_auto': False,
        'align_type': 'mse_margin',
        'alpha_user': 0.0,
        'alpha_item': 0.0,
        'gamma_user': 1.0,
        'gamma_item': 1.0,
        'beta_user': 0.0,
        'beta_item': 0.0,
        'g_beta_user': 1.0,
        'g_beta_item': 1.0,
        'disc_scale': 10,
        'pred_dnn_units': [256, 128],
        'pred_dropout_rate': 0.5,
        'pred_activation': 'sigmoid',
        'interact_type': 'irt_wo_disc',
    }

    def build_cfg(self):
        self.user_count = self.datatpl_cfg['dt_info']['stu_count']
        self.item_count = self.datatpl_cfg['dt_info']['exer_count']
        self.cpt_count = self.datatpl_cfg['dt_info']['cpt_count']

    def add_extra_data(self, **kwargs):
        self.Q_mat = kwargs['missing_Q_mat'].to(self.device)
        self.dict_cpt_relation = {
            k:torch.LongTensor(v).to(self.device) for k,v in kwargs['dict_cpt_relation'].items()
        }
        self.interact_mat = kwargs['interact_mat'].to(self.device).float()

    def build_model(self):
        self.EncoderUser = MLP(
            input_dim=self.item_count,
            output_dim=self.cpt_count * 2,
            dnn_units=self.modeltpl_cfg['EncoderUserHidden']
        )
        self.EncoderItem = MLP(
            input_dim=self.user_count,
            output_dim=self.cpt_count,
            dnn_units=self.modeltpl_cfg['EncoderItemHidden']
        )

        self.EncoderItemDiff = MLP(
            input_dim=self.user_count,
            output_dim=self.cpt_count * 2,
            dnn_units=self.modeltpl_cfg['EncoderItemHidden']
        )


        self.ItemDisc = nn.Embedding(self.item_count, 1)
        self.pd_net = PosMLP(
            input_dim=self.cpt_count, output_dim=1, activation=self.modeltpl_cfg['pred_activation'],
            dnn_units=self.modeltpl_cfg['pred_dnn_units'], dropout_rate=self.modeltpl_cfg['pred_dropout_rate']
        )

        self.margin_loss_zero_one = MarginLossZeroOne(reduction='none', margin=self.modeltpl_cfg['align_margin_loss_kwargs']['margin'])

        self.user_dist = NormalDistUtil()
        self.item_dist = BernoulliUtil(p=self.modeltpl_cfg['bernoulli_prior_p'], stgradient=True)
        self.item_dist_diff = NormalDistUtil()
    
    def get_align_item_loss(self, item_emb, item_idx):
        if self.modeltpl_cfg['align_type'] == 'mse_margin':
            flag = self.Q_mat[item_idx, :].sum(dim=1) > 0
            left_emb = item_emb[~flag]
            p = self.modeltpl_cfg['align_margin_loss_kwargs']['norm']
            t_loss = torch.norm(left_emb, dim=0, p=p).pow(p).sum()
            if left_emb.shape[0] != 0 and self.callback_list.curr_epoch >= self.modeltpl_cfg['align_margin_loss_kwargs']['start_epoch']:
                # topk_idx = torch.topk(left_emb, self.modeltpl_cfg['align_margin_loss_kwargs']['topk']).indices
                # bottomk_idx = torch.ones_like(left_emb).scatter(1, topk_idx, 0).nonzero()[:, 1].reshape(-1, left_emb.size(1) - topk_idx.size(1))
                # pos = torch.gather(left_emb, 1, topk_idx[:,[-1]])
                # neg = torch.gather(left_emb, 1, bottomk_idx[:,torch.randperm(bottomk_idx.shape[1],dtype=torch.long)[0:int(bottomk_idx.shape[1]*0.5)]])
                topk_idx = torch.topk(left_emb, self.modeltpl_cfg['align_margin_loss_kwargs']['topk']+1).indices
                pos = torch.gather(left_emb, 1, topk_idx[:,0:self.modeltpl_cfg['align_margin_loss_kwargs']['d1']])
                neg = torch.gather(left_emb, 1, topk_idx[:,[-1]])
                margin_loss = self.margin_loss_zero_one(pos, neg).mean(dim=1).sum()
            else:
                margin_loss = torch.tensor(0.0).to(self.device)
            return {
                "mse_loss": F.mse_loss(item_emb[flag], self.Q_mat[item_idx[flag], :].float(), reduction='sum'),
                "margin_loss": margin_loss,
                "norm_loss": t_loss,
            }
        elif self.modeltpl_cfg['align_type'] == 'mse_margin_mean':
            flag = self.Q_mat[item_idx, :].sum(dim=1) > 0
            left_emb = item_emb[~flag]
            p = self.modeltpl_cfg['align_margin_loss_kwargs']['norm']
            t_loss = torch.norm(left_emb, dim=0, p=p).pow(p).sum()
            if left_emb.shape[0] != 0 and self.callback_list.curr_epoch >= self.modeltpl_cfg['align_margin_loss_kwargs']['start_epoch']:
                # topk_idx = torch.topk(left_emb, self.modeltpl_cfg['align_margin_loss_kwargs']['topk']).indices
                # bottomk_idx = torch.ones_like(left_emb).scatter(1, topk_idx, 0).nonzero()[:, 1].reshape(-1, left_emb.size(1) - topk_idx.size(1))
                # pos = torch.gather(left_emb, 1, topk_idx[:,[-1]])
                # neg = torch.gather(left_emb, 1, bottomk_idx[:,torch.randperm(bottomk_idx.shape[1],dtype=torch.long)[0:int(bottomk_idx.shape[1]*0.5)]])
                topk_idx = torch.topk(left_emb, self.modeltpl_cfg['align_margin_loss_kwargs']['topk']+1).indices
                bottomk_idx = torch.topk(-left_emb, left_emb.shape[1] -  self.modeltpl_cfg['align_margin_loss_kwargs']['topk']).indices
                pos = torch.gather(left_emb, 1, topk_idx[:,0:self.modeltpl_cfg['align_margin_loss_kwargs']['d1']])
                neg = torch.gather(left_emb, 1, bottomk_idx).mean(dim=1)
                margin_loss = self.margin_loss_zero_one(pos, neg).mean(dim=1).sum()
            else:
                margin_loss = torch.tensor(0.0).to(self.device)
            return {
                "mse_loss": F.mse_loss(item_emb[flag], self.Q_mat[item_idx[flag], :].float(), reduction='sum'),
                "margin_loss": margin_loss,
                "norm_loss": t_loss,
            }
        else:
            raise ValueError(f"Unknown align type: {self.modeltpl_cfg['align_type']}")


    def decode(self, user_emb, item_emb, item_emb_diff, item_id, **kwargs):
        if self.modeltpl_cfg['interact_type'] == 'irt_wo_disc':
            return ((user_emb - item_emb_diff)*item_emb).sum(dim=1)
        elif self.modeltpl_cfg['interact_type'] == 'irt':
            item_disc = self.ItemDisc(item_id).sigmoid() #* self.modeltpl_cfg['disc_scale']
            return ((user_emb - item_emb_diff)*item_emb*item_disc).sum(dim=1)
        elif self.modeltpl_cfg['interact_type'] == 'ncdm':
            item_disc = self.ItemDisc(item_id).sigmoid()# * self.modeltpl_cfg['disc_scale']
            input = (user_emb - item_emb_diff)*item_emb*item_disc
            return self.pd_net(input).flatten()
        elif self.modeltpl_cfg['interact_type'] == 'mf':
            return ((user_emb.sigmoid()*item_emb)*(item_emb*item_emb_diff)).sum(dim=1)
        elif self.modeltpl_cfg['interact_type'] == 'mirt': # 就是mf加了个disc
            item_disc = self.ItemDisc(item_id).sigmoid() #* self.modeltpl_cfg['disc_scale']
            return ((user_emb.sigmoid()*item_emb)*(item_emb*item_emb_diff)).sum(dim=1) + item_disc.flatten()
        else:
            raise NotImplementedError

    def forward(self, users, items, labels):
        user_unique, user_unique_idx = users.unique(sorted=True, return_inverse=True)
        item_unique, item_unique_idx = items.unique(sorted=True, return_inverse=True)

        user_mix = self.EncoderUser(self.interact_mat[user_unique, :])
        user_mu, user_logvar = torch.chunk(user_mix, 2, dim=-1)
        user_emb_ = self.user_dist.sample(user_mu, user_logvar)
        user_emb = user_emb_[user_unique_idx, :]

        item_mu = self.EncoderItem(self.interact_mat[:, item_unique].T).sigmoid()
        item_emb_ = self.item_dist.sample(None, item_mu, type_=self.modeltpl_cfg['b_sample_type'], **self.modeltpl_cfg['b_sample_kwargs'])
        item_emb = item_emb_[item_unique_idx, :]


        item_diff_mix = self.EncoderItemDiff(self.interact_mat[:, item_unique].T)
        item_mu_diff, item_logvar_diff = torch.chunk(item_diff_mix, 2, dim=-1)
        item_emb_diff_ = self.item_dist_diff.sample(item_mu_diff, item_logvar_diff)
        item_emb_diff = item_emb_diff_[item_unique_idx, :]

        loss_main = F.binary_cross_entropy_with_logits(self.decode(user_emb, item_emb, item_emb_diff, item_id=items), labels, reduction='sum') # 重构 loss
        align_loss_dict = self.get_align_item_loss(item_mu, item_unique)
        # align_loss_dict_diff = self.get_align_item_loss(item_mu_diff, item_unique)

        user_terms = self.get_tcvae_terms(user_emb_, params=(user_mu, user_logvar), dist=self.user_dist, dataset_size=self.user_count)
        item_terms = self.get_tcvae_terms(item_emb_, params=item_mu, dist=self.item_dist, dataset_size=self.item_count)
        item_terms_diff = self.get_tcvae_terms(item_emb_diff_, params=(item_mu_diff, item_logvar_diff), dist=self.item_dist_diff, dataset_size=self.item_count)
        
        return {
            'loss_main': loss_main * self.modeltpl_cfg['lambda_main'],
            'loss_mse': align_loss_dict['mse_loss'] * self.modeltpl_cfg['lambda_q'],
            'loss_margin': align_loss_dict['margin_loss'] * self.modeltpl_cfg['align_margin_loss_kwargs']['margin_lambda'],
            'loss_norm': align_loss_dict['norm_loss'] * self.modeltpl_cfg['align_margin_loss_kwargs']['norm_lambda'],
            # 'loss_mse_diff': align_loss_dict_diff['mse_loss'] * self.modeltpl_cfg['lambda_q'],
            # 'loss_margin_diff': align_loss_dict_diff['margin_loss'] * self.modeltpl_cfg['align_margin_loss_kwargs']['margin_lambda'],
            # 'loss_norm_diff': align_loss_dict_diff['norm_loss'] * self.modeltpl_cfg['align_margin_loss_kwargs']['norm_lambda'],
            
            'user_MI': user_terms['MI'] * self.modeltpl_cfg['alpha_user'],
            'user_TC': user_terms['TC'] * self.modeltpl_cfg['beta_user'],
            'user_TC_G': user_terms['TC_G'] * self.modeltpl_cfg['g_beta_user'],
            'user_KL': user_terms['KL'] * self.modeltpl_cfg['gamma_user'],
            'item_MI': item_terms['MI'] * self.modeltpl_cfg['alpha_item'],
            'item_TC': item_terms['TC'] * self.modeltpl_cfg['beta_item'],
            'item_TC_G': item_terms['TC_G'] * self.modeltpl_cfg['g_beta_item'],
            'item_KL': item_terms['KL'] * self.modeltpl_cfg['gamma_item'],
            
            'item_MI_diff': item_terms_diff['MI'] * self.modeltpl_cfg['alpha_item'],
            'item_TC_diff': item_terms_diff['TC'] * self.modeltpl_cfg['beta_item'],
            'item_TC_G_diff': item_terms_diff['TC_G'] * self.modeltpl_cfg['g_beta_item'],
            'item_KL_diff': item_terms_diff['KL'] * self.modeltpl_cfg['gamma_item'],
        }

    def get_tcvae_terms(self, z, params, dist, dataset_size):
        batch_size, latent_dim = z.shape

        if isinstance(dist, NormalDistUtil):
            mu, logvar = params
            zero = torch.FloatTensor([0.0]).to(self.device)
            logpz = dist.log_density(X=z, MU=zero, LOGVAR=zero).sum(dim=1)
            logqz_condx = dist.log_density(X=z, MU=mu, LOGVAR=logvar).sum(dim=1)
            _logqz = dist.log_density(
                z.reshape(batch_size, 1, latent_dim),
                mu.reshape(1, batch_size, latent_dim),
                logvar.reshape(1, batch_size, latent_dim)
            ) # _logqz的第(i,j,k)个元素, P(z(n_i)_k|n_j)
        elif isinstance(dist, BernoulliUtil):
            logpz = dist.log_density(z, params=None).sum(dim=1)
            logqz_condx = dist.log_density(z, params=params).sum(dim=1)
            # _logqz = torch.stack([dist.log_density(z, params[i,:]) for i in range(batch_size)],dim=1)
            _logqz = dist.log_density(z.reshape(batch_size, 1, latent_dim), params=params.reshape(1, batch_size, latent_dim), is_check=False)
        else:
            raise ValueError("unknown base class of dist")

        if self.modeltpl_cfg['sampling_type'] == 'mws':
            # minibatch weighted sampling
            logqz_prodmarginals = (torch.logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (torch.logsumexp(_logqz.sum(dim=2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
            logqz_group_list = []
            if hasattr(self, 'dict_cpt_relation'):
                for gid, group_idx in self.dict_cpt_relation.items():
                    logqz_group_list.append(
                        (torch.logsumexp(_logqz[:,:,group_idx].sum(dim=2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
                    )
                logqz_group = torch.vstack(logqz_group_list).T.sum(dim=1)
        elif self.modeltpl_cfg['sampling_type'] == 'mss':
            logiw_mat = self._log_importance_weight_matrix(z.shape[0], dataset_size).to(z.device)
            logqz = torch.logsumexp(
                logiw_mat + _logqz.sum(dim=-1), dim=-1
            )  # MMS [B]
            logqz_prodmarginals = (
                torch.logsumexp(
                    logiw_mat.reshape(z.shape[0], z.shape[0], -1) + _logqz,
                    dim=1,
                )
            ).sum(
                dim=-1
            )
            logqz_group_list = []
            if hasattr(self, 'dict_cpt_relation'):
                for gid, group_idx in self.dict_cpt_relation.items():
                    logqz_group_list.append(
                       (
                        torch.logsumexp(
                            logiw_mat.reshape(z.shape[0], z.shape[0], -1) + _logqz[:,:,group_idx], dim=1,
                        )).sum(dim=-1)
                    )
                logqz_group = torch.vstack(logqz_group_list).T.sum(dim=1)

        else:
            raise ValueError("Unknown Sampling Type")
        
        IndexCodeMI = logqz_condx - logqz
        TC = logqz - logqz_prodmarginals
        TC_G = (logqz - logqz_group).mean() if hasattr(self, 'dict_cpt_relation') else torch.FloatTensor([0.0]).to(self.device)
        DW_KL = logqz_prodmarginals - logpz
        return {
            'MI': IndexCodeMI.mean(),
            'TC': TC.mean(),
            'TC_G': TC_G,
            'KL': DW_KL.mean()
        }


    @torch.no_grad()
    def predict(self, stu_id, exer_id, **kwargs):
        users = stu_id
        items = exer_id
        user_emb = None
        if users is None:
            user_mix = self.EncoderUser(self.interact_mat)
            user_emb, _ = torch.chunk(user_mix, 2, dim=-1)
        else:
            user_mix = self.EncoderUser(self.interact_mat[users, :])
            user_emb, _ = torch.chunk(user_mix, 2, dim=-1)

        item_emb = None
        if items is None:
            item_emb = self.EncoderItem(self.interact_mat.T).sigmoid()
        else:
            item_emb = self.EncoderItem(self.interact_mat[:, items].T).sigmoid()

        item_emb_diff = None
        if items is None:
            item_emb_diff_mix = self.EncoderItemDiff(self.interact_mat.T)
            item_emb_diff, _ = torch.chunk(item_emb_diff_mix, 2, dim=-1)
        else:
            item_emb_diff_mix = self.EncoderItemDiff(self.interact_mat[:, items].T)
            item_emb_diff, _ = torch.chunk(item_emb_diff_mix, 2, dim=-1)

        return {"y_pd":self.decode(user_emb, item_emb, item_emb_diff, item_id=items).sigmoid()}

    @torch.no_grad()
    def get_stu_status(self, users=None):
        user_emb = None
        if users is None:
            user_mix = self.EncoderUser(self.interact_mat)
            user_emb, _ = torch.chunk(user_mix, 2, dim=-1)
        else:
            user_mix = self.EncoderUser(self.interact_mat[users, :])
            user_emb, _ = torch.chunk(user_mix, 2, dim=-1)
        
        return user_emb

    @torch.no_grad()
    def get_exer_emb(self, items=None):
        item_emb = None
        if items is None:
            item_emb = self.EncoderItem(self.interact_mat.T)
        else:
            item_emb = self.EncoderItem(self.interact_mat[:, items].T)
        
        return item_emb.sigmoid()

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        """Compute importance weigth matrix for MSS
        Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
        """

        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def get_main_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        label = kwargs['label']
        return self(stu_id, exer_id, label)

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)



eps=1e-8


class STHeaviside(Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.zeros(x.size()).type_as(x)
        y[x >= 0] = 1
        return y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class NormalDistUtil(object):
    @staticmethod
    def log_density(X, MU, LOGVAR):
        """ compute log pdf of normal distribution

        Args:
            X (_type_): sample point
            MU (_type_): mu of normal dist
            LOGVAR (_type_): logvar of normal dist
        """
        norm = - 0.5 * (math.log(2 * math.pi) + LOGVAR)
        log_density = norm - 0.5 * ((X - MU).pow(2) * torch.exp(-LOGVAR))
        return log_density

    @staticmethod
    def kld(MU:float, LOGVAR:float, mu_move):
        """compute KL divergence between X and Normal Dist whose (mu, var) equals to (mu_move, 1)

        Args:
            MU (float): _description_
            VAR (float): _description_
            mu_move (_type_): _description_
        """

        return 0.5 * (LOGVAR.exp() - LOGVAR + MU.pow(2) - 2 * mu_move * MU + mu_move ** 2 - 1)

    @staticmethod
    def sample(mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps


class BernoulliUtil(nn.Module):
    """Samples from a Bernoulli distribution where the probability is given
    by the sigmoid of the given parameter.
    """

    def __init__(self, p=0.5, stgradient=False):
        super().__init__()
        p = torch.Tensor([p])
        self.p = torch.log(p / (1 - p) + eps)
        self.stgradient = stgradient

    def _check_inputs(self, size, ps):
        if size is None and ps is None:
            raise ValueError(
                'Either one of size or params should be provided.')
        elif size is not None and ps is not None:
            if ps.ndimension() > len(size):
                return ps.squeeze(-1).expand(size)
            else:
                return ps.expand(size)
        elif size is not None:
            return self.p.expand(size)
        elif ps is not None:
            return ps
        else:
            raise ValueError(
                'Given invalid inputs: size={}, ps={})'.format(size, ps))

    def _sample_logistic(self, size):
        u = torch.rand(size)
        l = torch.log(u + eps) - torch.log(1 - u + eps)
        return l

    def default_sample(self, size=None, params=None):
        presigm_ps = self._check_inputs(size, params)
        logp = F.logsigmoid(presigm_ps)
        logq = F.logsigmoid(-presigm_ps)
        l = self._sample_logistic(logp.size()).type_as(presigm_ps)
        z = logp - logq + l
        b = STHeaviside.apply(z)
        return b if self.stgradient else b.detach()

    def sample(self, size=None, params=None,type_='gumbel_softmax', **kwargs):
        if type_ == 'default':
            return self.default_sample(size, params)
        elif type_ == 'gumbel_softmax':
            tau = kwargs.get('tau', 1.0)
            hard = kwargs.get('hard', True)
            ext_params = torch.log(torch.stack([1 - params, params],dim=2) + eps)
            return F.gumbel_softmax(logits=ext_params, tau=tau, hard=hard)[:,:,-1]
        else:
            raise ValueError(f"Unknown Type of sample: {type_}")

    def log_density(self, sample, params=None, is_check=True):
        if is_check:
            presigm_ps = self._check_inputs(sample.size(), params).type_as(sample)
        else:
            presigm_ps = params
        p = (torch.sigmoid(presigm_ps) + eps) * (1 - 2 * eps)
        logp = sample * torch.log(p + eps) + (1 - sample) * torch.log(1 - p + eps)
        return logp

    def get_params(self):
        return self.p

    @property
    def nparams(self):
        return 1

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return self.stgradient

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f})'.format(
            torch.sigmoid(self.p.data)[0])
        return tmpstr
