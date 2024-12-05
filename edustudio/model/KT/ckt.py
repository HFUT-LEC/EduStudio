from ..gd_basemodel import GDBaseModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class CKT(GDBaseModel):
    default_cfg = {
        'hidden_size': 100,
        'drop_rate': 0,
        'k1': 6,
        'k2': 6
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def build_cfg(self):
        self.n_user = self.datatpl_cfg['dt_info']['stu_count']
        self.n_item = self.datatpl_cfg['dt_info']['exer_count']

        self.hidden_size = self.modeltpl_cfg['hidden_size']
        self.num_skills = self.n_item

    def build_model(self):
        skill_w = torch.empty(self.num_skills, self.hidden_size, device=self.device)
        torch.nn.init.xavier_uniform_(skill_w)
        self.skill_w = torch.nn.Parameter(skill_w)

        zeros = torch.zeros((self.num_skills, self.hidden_size), dtype=torch.float32, device=self.device)
        t1 = torch.cat([self.skill_w, zeros], dim=-1)
        t2 = torch.cat([zeros, self.skill_w], dim=-1)
        self.input_w = torch.cat([t1, t2], dim=0)

        self.cnn_block = CNNBlock(self.modeltpl_cfg['hidden_size'],
                                  self.modeltpl_cfg['k1'],
                                  self.modeltpl_cfg['k2'],
                                  self.modeltpl_cfg['drop_rate'],
                                  self.device).to(self.device)

        self.cnn = CNN(self.modeltpl_cfg['hidden_size'],
                       self.cnn_block,
                       self.device
                       ).to(self.device)

    def forward(self, exer_seq, label_seq, **kwargs):
        input_data, input_skill, l, next_id = self.data_helper(exer_seq, label_seq)
        skills = torch.nn.functional.embedding(input_skill.long(), self.skill_w)
        next_skill = torch.nn.functional.embedding(next_id.long(), self.skill_w)
        input_data = torch.nn.functional.embedding(input_data.long(), self.input_w)

        outputs = self.cnn(input_data, skills, l, next_skill)
        logits = torch.sum(outputs, dim=-1)
        y_pd = logits.sigmoid()
        return y_pd

    @torch.no_grad()
    def predict(self, **kwargs):
        y_pd = self(**kwargs)
        y_pd = y_pd[:, :-1][kwargs['mask_seq'][:, 1:] == 1]
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
        y_pd = y_pd[:, :-1][kwargs['mask_seq'][:, 1:] == 1]
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

    def data_helper(self, exer_seq, label_seq):
        num_steps = self.datatpl_cfg['dt_info']['real_window_size']
        batch_size = len(exer_seq)
        input_data = torch.zeros((batch_size, num_steps), device=self.device)
        input_skill = torch.zeros((batch_size, num_steps), device=self.device)
        next_id = torch.zeros((batch_size, num_steps), device=self.device)
        l = torch.ones((batch_size, num_steps, self.num_skills), device=self.device)
        for i in range(batch_size):

            problem_ids = exer_seq[i]

            correctness = label_seq[i]
            correct_num = torch.zeros(self.num_skills, device=self.device)
            answer_count = torch.ones(self.num_skills, device=self.device)
            for j in range(len(problem_ids) - 1):
                problem_id = int(problem_ids[j])

                if (int(correctness[j]) == 0):
                    input_data[i, j] = problem_id + self.num_skills
                else:
                    input_data[i, j] = problem_id
                    correct_num[problem_id] += 1
                l[i, j] = correct_num / answer_count
                answer_count[problem_id] += 1
                input_skill[i, j] = problem_id
                next_id[i, j] = int(problem_ids[j + 1])
        return input_data, input_skill, l, next_id


def GLU(inputs, dim, device):
    sigmoid = nn.Sigmoid().to(device)
    dense1 = nn.Linear(inputs.size(-1), dim).to(device)
    dense2 = nn.Linear(inputs.size(-1), dim).to(device)

    r = sigmoid(dense1(inputs))

    output = dense2(inputs)

    output = output * r
    return output


class CNNBlock(nn.Module):
    def __init__(self, filter1, k1, k2, drop_rate, device):
        super(CNNBlock, self).__init__()

        self.filter1 = filter1
        self.k1 = k1
        self.k2 = k2
        self.drop_rate = drop_rate
        self.device = device

        self.w1 = nn.Parameter(torch.Tensor(k1, filter1, filter1))
        nn.init.xavier_uniform_(self.w1)

        self.w2 = torch.zeros(k2, filter1, filter1)

        self.res_w = nn.Parameter(torch.cat([self.w1, self.w2], dim=0))

        self.b = nn.Parameter(torch.Tensor(filter1))
        nn.init.uniform_(self.b)


    def forward(self, x):
        o1 = x
        o2 = F.conv1d(o1.permute([0, 2, 1]), self.res_w.permute([2, 1, 0]), stride=1, padding='same').permute(
            [0, 2, 1]) + self.b
        o2 = F.dropout(o2, p=self.drop_rate)
        o2 = GLU(o2, self.filter1, self.device)
        return o2 + x


class CNN(nn.Module):
    def __init__(self, hidden_size, cnn_block, device):
        super(CNN, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        self.cnn_block = cnn_block

    def forward(self, x, skills, individual, next_skill):
        att = self.cal_att(skills)

        x2 = torch.matmul(att, x)

        x = torch.cat([x, x2, individual], axis=-1)
        x = GLU(x, self.hidden_size, self.device)
        # x = self.glu(x)
        for i in range(1, 4):
            x = self.cnn_block(x)
        x = x * next_skill
        return x

    def cal_att(self, inputs):
        xnorm = torch.sqrt(torch.sum(torch.square(inputs), axis=2))

        xnorm1 = (xnorm.unsqueeze(1)).repeat([1, inputs.shape[1], 1])
        xnorm2 = (xnorm.unsqueeze(-1)).repeat([1, 1, inputs.shape[1]])

        x_x = inputs.matmul(inputs.permute([0, 2, 1]))

        outputs = torch.div(x_x, xnorm1 * xnorm2)
        diag_vals = torch.ones_like(outputs[0, :, :], device=self.device)
        tril = torch.tril(diag_vals, diagonal=2)
        sel = torch.eye(outputs.shape[1], device=self.device)
        sel2 = torch.eye(1, m=outputs.shape[1], device=self.device)
        sel3 = torch.zeros([outputs.shape[1] - 1, outputs.shape[1]], device=self.device)
        sel4 = torch.cat([sel2, sel3], dim=0)
        sel = sel - sel4
        tril = tril - sel

        masks = tril.unsqueeze(0).repeat([outputs.shape[0], 1, 1])
        paddings = torch.ones_like(masks, device=self.device) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, paddings, outputs)

        outputs = F.softmax(outputs, dim=1)

        return outputs
