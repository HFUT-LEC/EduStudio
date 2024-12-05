from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F

class DKVMNHeadGroup(nn.Module):
    def forward(self, input_):
        pass

    def __init__(self, memory_size, memory_state_dim, is_write, device):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)
        self.device = device

    @staticmethod
    def addressing(control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = F.softmax(similarity_score, dim=1)  # Shape: (batch_size, memory_size)
        
        return correlation_weight
    
    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1) 
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)  
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)

        return read_content
    def write(self, control_input, memory, write_weight=None):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        if write_weight is None:
            write_weight = self.addressing(control_input=control_input, memory=memory)
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mul = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mul) + add_mul

        return new_memory

class DKVMN(nn.Module):
    def forward(self, input_):
        pass

    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key, device):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False,
                                       device=device)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True,
                                         device=device)
        self.memory_key = init_memory_key
        self.device = device

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        
        return correlation_weight

    def read(self, read_weight, memory_value):
        read_content = self.value_head.read(memory=memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input, memory_value):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=memory_value,
                                             write_weight=write_weight,
                                             )
        
        return memory_value

class SKVMN(GDBaseModel):
    default_cfg = {
        'memory_size': 50,
        'embed_dim': 200,  
        'dropout': 0.2,
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']
        self.window_size = self.datatpl_cfg['dt_info']['real_window_size']
        

    def build_model(self):
        self.x_emb_layer = nn.Embedding(2 * self.n_item, self.modeltpl_cfg['embed_dim'])
        self.k_emb_layer = nn.Embedding(self.n_item, self.modeltpl_cfg['embed_dim'])
        self.Mk = nn.Parameter(torch.Tensor(self.modeltpl_cfg['memory_size'], self.modeltpl_cfg['embed_dim']))
        self.Mv = nn.Parameter(torch.Tensor(self.modeltpl_cfg['memory_size'], self.modeltpl_cfg['embed_dim']))

        nn.init.kaiming_normal_(self.Mk)
        nn.init.kaiming_normal_(self.Mv)

        self.read_embed_linear = nn.Linear(
            2 * self.modeltpl_cfg['embed_dim'],
            self.modeltpl_cfg['embed_dim'],
            bias=True
        )
        self.a_embed = nn.Linear(
            2 * self.modeltpl_cfg['embed_dim'],
            self.modeltpl_cfg['embed_dim'],
            bias=True
            )
        self.mem = DKVMN(
            memory_size=self.modeltpl_cfg['memory_size'],
            memory_key_state_dim=self.modeltpl_cfg['embed_dim'],
            memory_value_state_dim=self.modeltpl_cfg['embed_dim'],
            init_memory_key=self.Mk,
            device=self.traintpl_cfg['device']
        )
        self.hx = nn.Parameter(torch.Tensor(1, self.modeltpl_cfg['embed_dim']))
        self.cx = nn.Parameter(torch.Tensor(1, self.modeltpl_cfg['embed_dim']))
        nn.init.kaiming_normal_(self.hx )
        nn.init.kaiming_normal_(self.cx)
        # modify hop_lstm
        self.lstm_cell = nn.LSTMCell(self.modeltpl_cfg['embed_dim'], self.modeltpl_cfg['embed_dim'])
        self.dropout_layer = nn.Dropout(self.modeltpl_cfg['dropout'])
        self.p_layer = nn.Linear(self.modeltpl_cfg['embed_dim'], 1, bias=True)

    def ut_mask(self, seq_len):
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=0).to(dtype=torch.bool)
    
    def triangular_layer(self, correlation_weight, seqlen=100, a=0.075, b=0.088, c=1.00):
        # w'= min((w-a)/(b-a), (c-w)/(c-b))
        # man(w', 0)
        correlation_weight = correlation_weight.view(self.bs * seqlen, -1)
        correlation_weight = torch.cat([correlation_weight[i] for i in range(correlation_weight.shape[0])], 0).unsqueeze(0)
        correlation_weight = torch.cat([(correlation_weight-a)/(b-a), (c-correlation_weight)/(c-b)], 0)
        correlation_weight, _ = torch.min(correlation_weight, 0)
        w0 = torch.zeros(correlation_weight.shape[0]).to(self.traintpl_cfg['device'])
        correlation_weight = torch.cat([correlation_weight.unsqueeze(0), w0.unsqueeze(0)], 0)
        correlation_weight, _ = torch.max(correlation_weight, 0)

        identity_vector_batch = torch.zeros(correlation_weight.shape[0]).to(self.traintpl_cfg['device'])
        # >=0.6 set to 2，0.1-0.6 set to 1，<=0.1 set to 0
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.lt(0.1), 0)
        identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.1), 1)
        _identity_vector_batch = identity_vector_batch.masked_fill(correlation_weight.ge(0.6), 2)

        identity_vector_batch = _identity_vector_batch.view(self.bs * seqlen, -1)
        identity_vector_batch = torch.reshape(identity_vector_batch,[self.bs, seqlen, -1])  #u(x): [bs, seqlen, size_m]
        # A^2
        iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=2, keepdim=True)
        iv_square_norm = iv_square_norm.repeat((1, 1, iv_square_norm.shape[1]))
        # B^2.T
        unique_iv_square_norm = torch.sum(torch.pow(identity_vector_batch, 2), dim=2, keepdim=True)
        unique_iv_square_norm = unique_iv_square_norm.repeat((1, 1, seqlen)).transpose(2, 1)
        # A * B.T
        iv_matrix_product = torch.bmm(identity_vector_batch, identity_vector_batch.transpose(2,1)) # A * A.T 
        # A^2 + B^2 - 2A*B.T
        iv_distances = iv_square_norm + unique_iv_square_norm - 2 * iv_matrix_product
        iv_distances = torch.where(iv_distances>0.0, torch.tensor(-1e32).to(self.traintpl_cfg['device']), iv_distances) 
        masks = self.ut_mask(iv_distances.shape[1]).to(self.traintpl_cfg['device'])
        mask_iv_distances = iv_distances.masked_fill(masks, value=torch.tensor(-1e32).to(self.traintpl_cfg['device'])) 
        idx_matrix = torch.arange(0,seqlen * seqlen,1).reshape(seqlen,-1).repeat(self.bs,1,1).to(self.traintpl_cfg['device'])
        final_iv_distance = mask_iv_distances + idx_matrix 
        values, indices = torch.topk(final_iv_distance, 1, dim=2, largest=True) 

        _values = values.permute(1,0,2)
        _indices = indices.permute(1,0,2)
        batch_identity_indices = (_values >= 0).nonzero() 
        identity_idx = []
        for identity_indices in batch_identity_indices:
            pre_idx = _indices[identity_indices[0],identity_indices[1]] 
            idx = torch.cat([identity_indices[:-1],pre_idx], dim=-1)
            identity_idx.append(idx)
        if len(identity_idx) > 0:
            identity_idx = torch.stack(identity_idx, dim=0)
        else:
            identity_idx = torch.tensor([]).to(self.traintpl_cfg['device'])

        return identity_idx 

    def forward(self, exer_seq, label_seq, **kwargs):
        k = self.k_emb_layer(exer_seq)  # Embedding Layer: about question
        self.bs = exer_seq.shape[0]
        x = (exer_seq + self.n_item * label_seq).long()

        value_read_content_l = []  # seqlen * bs * dim | read_content
        input_embed_l = []  # seqlen * bs * dim| q
        correlation_weight_list = []  # seqlen * bs * mem_size | correlation_weight
        ft = []
        # After atten about new question, Update Memory-value
        mem_value = self.Mv.unsqueeze(0).repeat(self.bs, 1, 1).to(self.traintpl_cfg['device'])  # bs * mem_size * dim
        for i in range(self.window_size):
            # k: bs * seqlen * dim
            q = k.permute(1, 0, 2)[i]  # q: bs * dim
            # attention Process
            correlation_weight = self.mem.attention(q).to(self.traintpl_cfg['device'])  # bs * memory_size
            
            # Read Process
            read_content = self.mem.read(correlation_weight, mem_value)  # read_count: bs * dim
            # modify
            batch_predict_input = torch.cat([read_content, q], 1)  #  bs * 2dim
            f = torch.tanh(self.read_embed_linear(batch_predict_input))

            # save intermedium data
            correlation_weight_list.append(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            ft.append(f)

            # Write Process  |  student response + f-vector 
            y = self.x_emb_layer(x[:,i])
            write_embed = torch.cat([f, y], 1) # bz * (f_dim + y_dim)
            write_embed = self.a_embed(write_embed).to(self.traintpl_cfg['device']) # bs * dim
            mem_value = self.mem.write(correlation_weight, write_embed, mem_value)
        # w: bs * seqlen * mem_size
        w = torch.cat([correlation_weight_list[i].unsqueeze(1) for i in range(self.window_size)], 1)
        idx_values = self.triangular_layer(w, self.window_size)
        ft = torch.stack(ft, dim=0)

        hidden_state, cell_state = [], []
        hx, cx = self.hx.repeat(self.bs, 1), self.cx.repeat(self.bs, 1)
        for i in range(self.window_size):
            for j in range(self.bs):
                if idx_values.shape[0] != 0 and i == idx_values[0][0] and j == idx_values[0][1]:
                    hx[j,:] = hidden_state[idx_values[0][2]][j]
                    cx = cx.clone()
                    cx[j,:] = cell_state[idx_values[0][2]][j]
                    idx_values = idx_values[1:]
            hx, cx = self.lstm_cell(ft[i], (hx, cx)) 
            hidden_state.append(hx) 
            cell_state.append(cx) 
        hidden_state = torch.stack(hidden_state, dim=0).permute(1,0,2)
        cell_state = torch.stack(cell_state, dim=0).permute(1,0,2)

        p = self.p_layer(self.dropout_layer(hidden_state))
        p = torch.sigmoid(p)
        y_pred = p.squeeze(-1)

        return y_pred
    
    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1]
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
        y_pd = y_pd[:, :-1]
        y_pd = y_pd[kwargs['mask_seq'][:, 1:] == 1]
        y_gt = kwargs['label_seq'][:, 1:]
        y_gt = y_gt[kwargs['mask_seq'][:, 1:] == 1]
        loss = F.binary_cross_entropy(
            input=y_pd, target=y_gt
        )
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        return self.get_main_loss(**kwargs)
