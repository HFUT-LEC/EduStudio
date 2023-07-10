from torch.utils.data import DataLoader
import random
from ..common import BaseProxyDataTPL
import numpy

class PairDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    
    def __iter__(self):
        iter_obj = super().__iter__()
        self.dataset.sample()
        return iter_obj



class IRRDataTPL(BaseProxyDataTPL):
    default_cfg = {
        'backbone_datatpl_cls': 'CDInterDataTPL',
        'num_observed': 10,
        'num_unobserved': 0
    }

    def sample(self):
        self.stu_count = self.datatpl_cfg['dt_info']['stu_count']
        self.num_observed = self.datatpl_cfg['num_observed']
        self.num_unobserved = self.datatpl_cfg['num_unobserved']
        stu_id = self.dict_main['stu_id'].numpy()
        exer_id = self.dict_main['exer_id'].numpy()
        label = self.dict_main['label'].numpy()
        students_with_correct_answers = {}
        students_with_wrong_answers = {}

        for i in range(stu_id.shape[0]):
            if label[i] == 0:
                students_with_wrong_answers.setdefault(exer_id[i], []).append(stu_id[i])
            else:
                students_with_correct_answers.setdefault(exer_id[i], []).append(stu_id[i])

        self.dict_main['pair_exer'] = []
        self.dict_main['pair_pos_stu'] = []
        self.dict_main['pair_neg_stu'] = []

        for i in range(stu_id.shape[0]):
            self.pair_exer = exer_id[i]

            if label[i] == 0:
                self.pair_neg_stu = stu_id[i]
                do_right_stu_exer = students_with_correct_answers.get(self.pair_exer, [])
                do_right_stu_exer_length = len(do_right_stu_exer)

                if do_right_stu_exer_length < self.num_observed:
                    for j in range(do_right_stu_exer_length):
                        self.add_pair(self.pair_exer, do_right_stu_exer[j], self.pair_neg_stu)
                else:
                    pos_stu_list = random.sample(do_right_stu_exer, k=self.num_observed)
                    for j in range(self.num_observed):
                        self.add_pair(self.pair_exer, pos_stu_list[j], self.pair_neg_stu)

                unobserved = []
                do_wrong_stu_exer = students_with_wrong_answers.get(self.pair_exer, [])

                while len(unobserved) < self.num_unobserved and len(unobserved) < do_right_stu_exer_length:
                    self.pair_pos_stu = numpy.int64(random.randint(0, self.stu_count - 1))

                    while (self.pair_pos_stu in do_right_stu_exer) or (self.pair_pos_stu in do_wrong_stu_exer) or (self.pair_pos_stu in unobserved):
                        self.pair_pos_stu = numpy.int64(random.randint(0, self.stu_count - 1))

                    unobserved.append(self.pair_pos_stu)

                for j in range(len(unobserved)):
                    self.add_pair(self.pair_exer, unobserved[j], self.pair_neg_stu)
            else:
                self.pair_pos_stu = stu_id[i]
                do_wrong_stu_exer = students_with_wrong_answers.get(self.pair_exer, [])
                do_wrong_stu_exer_length = len(do_wrong_stu_exer)

                if do_wrong_stu_exer_length < self.num_observed:
                    for j in range(do_wrong_stu_exer_length):
                        self.add_pair(self.pair_exer, self.pair_pos_stu, do_wrong_stu_exer[j])
                else:
                    neg_stu_list = random.sample(do_wrong_stu_exer, k=self.num_observed)
                    for j in range(self.num_observed):
                        self.add_pair(self.pair_exer, self.pair_pos_stu, neg_stu_list[j])

                unobserved = []
                do_right_stu_exer = students_with_correct_answers.get(self.pair_exer, [])

                while len(unobserved) < self.num_unobserved and len(unobserved) < do_wrong_stu_exer_length:
                    self.pair_neg_stu = numpy.int64(random.randint(0, self.stu_count - 1))

                    while (self.pair_neg_stu in do_right_stu_exer) or (self.pair_neg_stu in do_wrong_stu_exer) or (self.pair_neg_stu in unobserved):
                        self.pair_neg_stu = numpy.int64(random.randint(0, self.stu_count - 1))

                    unobserved.append(self.pair_neg_stu)

                for j in range(len(unobserved)):
                    self.add_pair(self.pair_exer, self.pair_pos_stu, unobserved[j])

    def add_pair(self, exer_id, pos_stu, neg_stu):
        self.dict_main['pair_exer'].append(exer_id)
        self.dict_main['pair_pos_stu'].append(pos_stu)
        self.dict_main['pair_neg_stu'].append(neg_stu)


    def build_dataloaders(self):
        batch_size = self.traintpl_cfg['batch_size']
        num_workers = self.traintpl_cfg['num_workers']
        eval_batch_size = self.traintpl_cfg['eval_batch_size']
        train_dt_list, valid_dt_list, test_dt_list = self.build_datasets()
        train_loader_list, valid_loader_list, test_loader_list = [], [], []

        for fid in range(self.datatpl_cfg['n_folds']):
            train_loader = PairDataLoader(dataset=train_dt_list[fid], shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
            train_loader_list.append(train_loader)
            if self.hasValidDataset:
                valid_loader = PairDataLoader(dataset=valid_dt_list[fid], shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
                valid_loader_list.append(valid_loader)
            test_loader = PairDataLoader(dataset=test_dt_list[fid], shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
            test_loader_list.append(test_loader)
        
        return train_loader_list, valid_loader_list, test_loader_list
