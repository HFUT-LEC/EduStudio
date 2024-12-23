r"""
CT_NCM
##################################
Reference:
    Haiping Ma et al. "Reconciling cognitive modeling with knowledge forgetting: A continuous time-aware neural network approach." in IJCAI 2022.
Reference code:
    https://github.com/BIMK/Intelligent-Education/tree/main/CTNCM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..gd_basemodel import GDBaseModel


class CT_NCM(GDBaseModel):
    """
    hidden_size: dimensions of LSTM hidden layerd
    embed_size: dimensions of student-knowledge concept interaction embedding
    prelen1: the first layer of performance prediction
    prelen2: the second layer of performance prediction
    dropout1: the proportion of first fully connected layer dropout before getting the prediction score
    dropout2: the proportion of second fully connected layer dropout before getting the prediction score
    """
    default_cfg = {
        'hidden_size': 64,
        'embed_size': 64,
        'prelen1': 256,
        'prelen2': 128,
        'dropout1': 0,
        'dropout2': 0,
    }

    def __init__(self, cfg):
        """Pass parameters from other templates into the model

        Args:
            cfg (UnifyConfig): parameters from other templates
        """
        super().__init__(cfg)

    def build_cfg(self):
        """Initialize the parameters of the model"""
        self.problem_num = self.datatpl_cfg['dt_info']['exer_count']
        self.skill_num = self.datatpl_cfg['dt_info']['cpt_count']
        self.device = self.traintpl_cfg['device']
        self.hidden_size = self.modeltpl_cfg['hidden_size']
        self.embed_size = self.modeltpl_cfg['embed_size']
        self.knowledge_dim = self.hidden_size
        self.input_len = self.knowledge_dim
        self.prelen1 = self.modeltpl_cfg['prelen1']
        self.prelen2 = self.modeltpl_cfg['prelen2']
        # 使用 BCEWithLogitsLoss，因为模型输出未经过 Sigmoid 激活
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        # 如果选择使用 BCELoss，请取消上面一行并启用下面一行
        # self.loss_function = torch.nn.BCELoss()

    def build_model(self):
        """Initialize the various components of the model"""
        self.dropout1 = nn.Dropout(p=self.modeltpl_cfg['dropout1'])
        self.dropout2 = nn.Dropout(p=self.modeltpl_cfg['dropout2'])

        self.inter_embedding = torch.nn.Embedding(2 * self.skill_num, self.embed_size)
        self.reclstm = torch.nn.Linear(self.embed_size + self.hidden_size, 7 * self.hidden_size)

        self.problem_disc = torch.nn.Embedding(self.problem_num, 1)
        self.problem_diff = torch.nn.Embedding(self.problem_num, self.knowledge_dim)

        self.linear1 = torch.nn.Linear(self.input_len, self.prelen1)
        self.linear2 = torch.nn.Linear(self.prelen1, self.prelen2)
        self.linear3 = torch.nn.Linear(self.prelen2, 1)

        # 手动初始化权重
        nn.init.xavier_normal_(self.reclstm.weight)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

    def forward(self, exer_seq, start_timestamp_seq, cpt_unfold_seq, label_seq, mask_seq, **kwargs):
        """A function of how well the model predicts students' responses to exercise questions

        Args:
            exer_seq (torch.Tensor): Sequence of exercise id. Shape of [batch_size, seq_len]
            start_timestamp_seq (torch.Tensor): Sequence of Students start answering time. Shape of [batch_size, seq_len]
            cpt_unfold_seq (torch.Tensor): Sequence of knowledge concepts related to exercises. Shape of [batch_size, seq_len]
            label_seq (torch.Tensor): Sequence of students' answers to exercises. Shape of [batch_size, seq_len]
            mask_seq (torch.Tensor): Sequence of mask. Mask=1 indicates that the student has answered the exercise, otherwise vice versa. Shape of [batch_size, seq_len]

        Returns:
            dict: The predictions of the model and the real situation
        """
        # 检查输入数据中是否存在 NaN 或 Inf
        assert not torch.isnan(exer_seq).any(), "exer_seq contains NaNs"
        assert not torch.isnan(start_timestamp_seq).any(), "start_timestamp_seq contains NaNs"
        assert not torch.isnan(cpt_unfold_seq).any(), "cpt_unfold_seq contains NaNs"
        assert not torch.isnan(label_seq).any(), "label_seq contains NaNs"
        assert not torch.isnan(mask_seq).any(), "mask_seq contains NaNs"
        assert not torch.isinf(exer_seq).any(), "exer_seq contains Inf"
        assert not torch.isinf(start_timestamp_seq).any(), "start_timestamp_seq contains Inf"
        assert not torch.isinf(cpt_unfold_seq).any(), "cpt_unfold_seq contains Inf"
        assert not torch.isinf(label_seq).any(), "label_seq contains Inf"
        assert not torch.isinf(mask_seq).any(), "mask_seq contains Inf"

        problem_seqs_tensor = exer_seq[:, 1:].to(self.device)
        skill_seqs_tensor = cpt_unfold_seq.to(self.device)
        start_timestamp_seqs_tensor = start_timestamp_seq[:, 1:].to(self.device)
        correct_seqs_tensor = label_seq.to(self.device)
        mask_labels = mask_seq.long().to(self.device)
        seqs_length = torch.sum(mask_labels, dim=1)
        delete_row = 0
        for i in range(len(seqs_length)):
            if seqs_length[i] == 1:
                mask = torch.arange(problem_seqs_tensor.size(0)) != (i - delete_row)
                problem_seqs_tensor = problem_seqs_tensor[mask]
                skill_seqs_tensor = skill_seqs_tensor[mask]
                start_timestamp_seqs_tensor = start_timestamp_seqs_tensor[mask]
                correct_seqs_tensor = correct_seqs_tensor[mask]
                mask_labels = mask_labels[mask]
                delete_row += 1

        # 将 mask_labels == 0 的标签设置为 0 而不是 -1
        correct_seqs_tensor = torch.where(mask_labels == 0, 0, correct_seqs_tensor)
        skill_seqs_tensor = torch.where(mask_labels == 0, 0, skill_seqs_tensor)
        mask_labels_temp = mask_labels[:, 1:]
        start_timestamp_seqs_tensor = torch.where(mask_labels_temp == 0, 0, start_timestamp_seqs_tensor)
        problem_seqs_tensor = torch.where(mask_labels_temp == 0, 0, problem_seqs_tensor)
        seqs_length = torch.sum(mask_labels, dim=1)

        # 再次检查处理后的数据
        assert not torch.isnan(problem_seqs_tensor).any(), "problem_seqs_tensor contains NaNs after processing"
        assert not torch.isnan(skill_seqs_tensor).any(), "skill_seqs_tensor contains NaNs after processing"
        assert not torch.isnan(start_timestamp_seqs_tensor).any(), "start_timestamp_seqs_tensor contains NaNs after processing"
        assert not torch.isnan(correct_seqs_tensor).any(), "correct_seqs_tensor contains NaNs after processing"

        inter_embed_tensor = self.inter_embedding(skill_seqs_tensor + self.skill_num * mask_labels)
        batch_size = correct_seqs_tensor.size()[0]

        hidden, _ = self.continues_lstm(inter_embed_tensor, start_timestamp_seqs_tensor, seqs_length, batch_size)
        hidden_packed = torch.nn.utils.rnn.pack_padded_sequence(hidden[1:, ],
                                                                seqs_length.cpu() - 1,
                                                                batch_first=False,
                                                                enforce_sorted=False)
        theta = hidden_packed.data
        problem_packed = torch.nn.utils.rnn.pack_padded_sequence(problem_seqs_tensor,
                                                                seqs_length.cpu() - 1,
                                                                batch_first=True,
                                                                enforce_sorted=False)
        predictions = torch.squeeze(self.problem_hidden(theta, problem_packed.data))
        labels_packed = torch.nn.utils.rnn.pack_padded_sequence(correct_seqs_tensor[:, 1:],
                                                                seqs_length.cpu() - 1,
                                                                batch_first=True,
                                                                enforce_sorted=False)
        labels = labels_packed.data
        out_dict = {'predictions': predictions, 'labels': labels}

        # 检查模型输出是否包含 NaN 或 Inf
        if torch.isnan(predictions).any():
            print("Predictions contain NaNs")
        if torch.isnan(labels).any():
            print("Labels contain NaNs")
        if torch.isinf(predictions).any():
            print("Predictions contain Inf")
        if torch.isinf(labels).any():
            print("Labels contain Inf")

        assert not torch.isnan(predictions).any(), "Predictions contain NaNs"
        assert not torch.isnan(labels).any(), "Labels contain NaNs"
        assert not torch.isinf(predictions).any(), "Predictions contain Inf"
        assert not torch.isinf(labels).any(), "Labels contain Inf"

        return out_dict

    def continues_lstm(self, inter_embed_tensor, start_timestamp_seqs_tensor, seqs_length, batch_size):
        """

        Args:
            inter_embed_tensor (torch.Tensor): interrelated LSTM unit. Shape of [batch_size, seq_len, embed_size]
            start_timestamp_seqs_tensor (torch.Tensor): Sequence of Students start answering time. Shape of [batch_size, seq_len-1]
            seqs_length (torch.Tensor): Length of sequence. Shape of [batch_size]
            batch_size (int): batch size.

        Returns:
            torch.Tensor: Output of LSTM.
        """
        self.init_states(batch_size=batch_size)
        h_list = [self.h_delay]
        for t in range(max(seqs_length) - 1):
            one_batch = inter_embed_tensor[:, t]
            c, self.c_bar, output_t, delay_t = self.conti_lstm(one_batch, self.h_delay, self.c_delay, self.c_bar)
            time_lag_batch = start_timestamp_seqs_tensor[:, t]
            self.c_delay, self.h_delay = self.delay(c, self.c_bar, output_t, delay_t, time_lag_batch)
            # 确保 h_delay 没有 NaN 或 Inf
            self.h_delay = torch.as_tensor(self.h_delay, dtype=torch.float).to(self.device)
            assert not torch.isnan(self.h_delay).any(), f"h_delay at time {t} contains NaNs"
            assert not torch.isinf(self.h_delay).any(), f"h_delay at time {t} contains Inf"
            h_list.append(self.h_delay)
        hidden = torch.stack(h_list)

        return hidden, seqs_length

    def init_states(self, batch_size):
        """Initialize the state of lstm

        Args:
            batch_size (int): batch_size
        """
        self.h_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_delay = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c_bar = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)
        self.c = torch.full((batch_size, self.hidden_size), 0.5, dtype=torch.float).to(self.device)

    def conti_lstm(self, one_batch_inter_embed, h_d_t, c_d_t, c_bar_t):
        """

        Args:
            one_batch_inter_embed (torch.Tensor): one batch of interrelated LSTM unit. Shape of [batch_size, embed_size]
            h_d_t (torch.Tensor): Shape of [batch_size, embed_size]
            c_d_t (torch.Tensor): Shape of [batch_size, embed_size]
            c_bar_t (torch.Tensor): Shape of [batch_size, embed_size]

        Returns:
            torch.Tensor: Data inside LSTM
        """
        input = torch.cat((one_batch_inter_embed, h_d_t), dim=1)
        (i, f, z, o, i_bar, f_bar, delay) = torch.chunk(self.reclstm(input), 7, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        z = torch.tanh(z)
        o = torch.sigmoid(o)
        i_bar = torch.sigmoid(i_bar)
        f_bar = torch.sigmoid(f_bar)
        delay = F.softplus(delay)

        # 检查中间变量是否包含 NaN 或 Inf
        assert not torch.isnan(i).any(), "i contains NaNs"
        assert not torch.isnan(f).any(), "f contains NaNs"
        assert not torch.isnan(z).any(), "z contains NaNs"
        assert not torch.isnan(o).any(), "o contains NaNs"
        assert not torch.isnan(i_bar).any(), "i_bar contains NaNs"
        assert not torch.isnan(f_bar).any(), "f_bar contains NaNs"
        assert not torch.isnan(delay).any(), "delay contains NaNs"
        assert not torch.isinf(i).any(), "i contains Inf"
        assert not torch.isinf(f).any(), "f contains Inf"
        assert not torch.isinf(z).any(), "z contains Inf"
        assert not torch.isinf(o).any(), "o contains Inf"
        assert not torch.isinf(i_bar).any(), "i_bar contains Inf"
        assert not torch.isinf(f_bar).any(), "f_bar contains Inf"
        assert not torch.isinf(delay).any(), "delay contains Inf"

        c_t = f * c_d_t + i * z
        c_bar_t = f_bar * c_bar_t + i_bar * z

        # 检查 c_t 和 c_bar_t 是否包含 NaN 或 Inf
        assert not torch.isnan(c_t).any(), "c_t contains NaNs"
        assert not torch.isnan(c_bar_t).any(), "c_bar_t contains NaNs"
        assert not torch.isinf(c_t).any(), "c_t contains Inf"
        assert not torch.isinf(c_bar_t).any(), "c_bar_t contains Inf"

        return c_t, c_bar_t, o, delay

    def delay(self, c, c_bar, output, delay, time_lag):
        """
        Args:
            c (torch.Tensor): Shape of [batch_size, embed_size]
            c_bar (torch.Tensor): Shape of [batch_size, embed_size]
            output (torch.Tensor): Shape of [batch_size, embed_size]
            delay (torch.Tensor): Shape of [batch_size, embed_size]
            time_lag (torch.Tensor): Shape of [batch_size]

        Returns:
            torch.Tensor: Data inside LSTM
        """
        exponent = - delay * time_lag.unsqueeze(-1)
        exponent = torch.clamp(exponent, min=-20, max=10)  # 限制指数的范围，max 降低到 10
        delta = c - c_bar
        delta = torch.clamp(delta, min=-10, max=10)  # 限制 delta 的范围
        c_delay = c_bar + delta * torch.exp(exponent)
        h_delay = output * torch.tanh(c_delay)

        # 检查 c_delay 和 h_delay 是否包含 NaN 或 Inf
        assert not torch.isnan(c_delay).any(), "c_delay contains NaNs"
        assert not torch.isnan(h_delay).any(), "h_delay contains NaNs"
        assert not torch.isinf(c_delay).any(), "c_delay contains Inf"
        assert not torch.isinf(h_delay).any(), "h_delay contains Inf"

        return c_delay, h_delay

    def problem_hidden(self, theta, problem_data):
        """Get how well the model predicts students' responses to exercise questions

        Args:
            theta (torch.Tensor): Student's ability value. Shape of [exer_num, seq_len]
            problem_data (torch.Tensor): The id of the exercise that the student has answered. Shape of [exer_num]

        Returns:
            torch.Tensor: the model predictions of students' responses to exercise questions. Shape of [exer_num, 1]
        """
        problem_diff = torch.sigmoid(self.problem_diff(problem_data))
        problem_disc = torch.sigmoid(self.problem_disc(problem_data))
        input_x = (theta - problem_diff) * problem_disc  # 移除 * 10

        # 添加限制
        input_x = torch.clamp(input_x, min=-10, max=10)  # 根据数据情况调整范围

        # 检查 input_x 是否包含 NaN 或 Inf
        assert not torch.isnan(input_x).any(), "input_x contains NaNs"
        assert not torch.isinf(input_x).any(), "input_x contains Inf"

        # 使用 ReLU 替代 Sigmoid
        input_x = self.dropout1(F.relu(self.linear1(input_x)))

        # 检查 input_x 是否包含 NaN 或 Inf
        assert not torch.isnan(input_x).any(), "input_x after linear1 and ReLU contains NaNs"
        assert not torch.isinf(input_x).any(), "input_x after linear1 and ReLU contains Inf"

        # 使用 ReLU 替代 Sigmoid
        input_x = self.dropout2(F.relu(self.linear2(input_x)))

        # 检查 input_x 是否包含 NaN 或 Inf
        assert not torch.isnan(input_x).any(), "input_x after linear2 and ReLU contains NaNs"
        assert not torch.isinf(input_x).any(), "input_x after linear2 and ReLU contains Inf"

        output = self.linear3(input_x)  # 移除 Sigmoid 激活

        # 检查 output 是否包含 NaN 或 Inf
        assert not torch.isnan(output).any(), "output contains NaNs"
        assert not torch.isinf(output).any(), "output contains Inf"

        # 可选：打印 output 的统计信息
        # print(f"Output - min: {output.min().item()}, max: {output.max().item()}")

        return output

    def predict(self, **kwargs):
        """A function of get how well the model predicts students' responses to exercise questions and the groundtruth

        Returns:
            dict: The predictions of the model and the real situation
        """
        outdict = self(**kwargs)
        # 在预测阶段应用 Sigmoid 激活以获取概率值
        y_pd = torch.sigmoid(outdict['predictions'])
        return {
            'y_pd': y_pd,
            'y_gt': torch.as_tensor(outdict['labels'], dtype=torch.float)
        }

    def get_main_loss(self, **kwargs):
        """

        Returns:
            dict: loss dict{'loss_main': loss_value}
        """
        outdict = self(**kwargs)
        predictions = outdict['predictions']
        labels = outdict['labels']
        labels = torch.as_tensor(labels, dtype=torch.float)

        # 添加断言以确保标签在 [0, 1] 范围内
        assert torch.all((labels == 0) | (labels == 1)), "标签中存在非 0 或 1 的值"

        # 检查 predictions 是否包含 NaN 或 Inf
        assert not torch.isnan(predictions).any(), "Predictions contain NaNs"
        assert not torch.isinf(predictions).any(), "Predictions contain Inf"

        loss = self.loss_function(predictions, labels)
        return {
            'loss_main': loss
        }

    def get_loss_dict(self, **kwargs):
        """

        Returns:
            dict: loss dict{'loss_main': loss_value}
        """
        return self.get_main_loss(**kwargs)
