from torch.utils.data import DataLoader
import random
from ..base_datafmt import BaseProxyDataFmt
import numpy

class PairDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def __iter__(self):
        iter_obj = super().__iter__()
        self.dataset.sample()
        return iter_obj



class IRRDataFmt(BaseProxyDataFmt):
    default_cfg = {
        'backbone_datafmt_cls': 'CDInterDataFmt',
        'num_observed': 10,
        'num_unobserved': 10
    }

    def sample(self):
        self.num_observed = self.datafmt_cfg['num_observed']
        self.num_unobserved = self.datafmt_cfg['num_unobserved']
        stu_id = self.dict_data['stu_id'].numpy()
        exer_id = self.dict_data['exer_id'].numpy()
        label = self.dict_data['label'].numpy()
        do_right_stu = {}
        do_wrong_stu = {}
        for i in range(stu_id.shape[0]):
            if label[i] == 0:
                if exer_id[i] in do_wrong_stu:
                    do_wrong_stu[exer_id[i]].append(stu_id[i])
                else:
                    do_wrong_stu[exer_id[i]] = [stu_id[i]]
            else:
                if exer_id[i] in do_right_stu:
                    do_right_stu[exer_id[i]].append(stu_id[i])
                else:
                    do_right_stu[exer_id[i]] = [stu_id[i]]
        self.dict_data['pair_exer'] = []
        self.dict_data['pair_pos_stu'] = []
        self.dict_data['pair_neg_stu'] = []
        for i in range(stu_id.shape[0]):
            self.pair_exer = exer_id[i]
            if label[i] == 0:
                self.pair_neg_stu = stu_id[i]
                do_right_stu_exer = []
                if self.pair_exer in do_right_stu:
                    do_right_stu_exer = do_right_stu[self.pair_exer]
                    if len(do_right_stu[self.pair_exer])<self.num_observed:
                        for j in range(len(do_right_stu[self.pair_exer])):
                            self.dict_data['pair_exer'].append(self.pair_exer)
                            self.dict_data['pair_pos_stu'].append(do_right_stu[self.pair_exer][j])
                            self.dict_data['pair_neg_stu'].append(self.pair_neg_stu)
                    else:
                        pos_stu_list = random.sample(do_right_stu[self.pair_exer], k=self.num_observed)
                        for j in range(self.num_observed):
                            self.dict_data['pair_exer'].append(self.pair_exer)
                            self.dict_data['pair_pos_stu'].append(pos_stu_list[j])
                            self.dict_data['pair_neg_stu'].append(self.pair_neg_stu)
                for j in range(self.num_unobserved):
                    self.pair_pos_stu = random.randint(0, self.stu_count - 1)
                    while self.pair_pos_stu in do_right_stu_exer:
                         self.pair_pos_stu = random.randint(0, self.stu_count - 1)
                    self.dict_data['pair_exer'].append(self.pair_exer)
                    self.dict_data['pair_pos_stu'].append(numpy.int64(self.pair_pos_stu))
                    self.dict_data['pair_neg_stu'].append(self.pair_neg_stu)
            else:
                self.pair_pos_stu = stu_id[i]
                do_wrong_stu_exer = []
                if self.pair_exer in do_wrong_stu:
                    do_wrong_stu_exer = do_wrong_stu[self.pair_exer]
                    if len(do_wrong_stu[self.pair_exer])<self.num_observed:
                        for j in range(len(do_wrong_stu[self.pair_exer])):
                            self.dict_data['pair_exer'].append(self.pair_exer)
                            self.dict_data['pair_pos_stu'].append(self.pair_pos_stu)
                            self.dict_data['pair_neg_stu'].append(do_wrong_stu[self.pair_exer][j])
                    else:
                        neg_stu_list = random.sample(do_wrong_stu[self.pair_exer], k=self.num_observed)
                        for j in range(self.num_observed):
                            self.dict_data['pair_exer'].append(self.pair_exer)
                            self.dict_data['pair_pos_stu'].append(self.pair_pos_stu)
                            self.dict_data['pair_neg_stu'].append(neg_stu_list[j])
                for j in range(self.num_unobserved):
                    self.pair_neg_stu = random.randint(0, self.stu_count - 1)
                    while self.pair_neg_stu in do_wrong_stu_exer:
                         self.pair_neg_stu = random.randint(0, self.stu_count - 1)
                    self.dict_data['pair_exer'].append(self.pair_exer)
                    self.dict_data['pair_pos_stu'].append(self.pair_pos_stu)
                    self.dict_data['pair_neg_stu'].append(numpy.int64(self.pair_neg_stu))

    def build_dataloaders(self):
        batch_size = self.trainfmt_cfg['batch_size']
        num_workers = self.trainfmt_cfg['num_workers']
        eval_batch_size = self.trainfmt_cfg['eval_batch_size']
        train_dataset, val_dataset, test_dataset = self.build_datasets()
        train_loader = PairDataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        if val_dataset is not None:
            val_loader = PairDataLoader(dataset=val_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        test_loader = PairDataLoader(dataset=test_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        return train_loader, val_loader, test_loader
