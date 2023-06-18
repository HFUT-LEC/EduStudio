from ..base_datafmt import BaseProxyDataFmt
import torch
import numpy as np
import pandas as pd


class DKTDSCDataFmt(BaseProxyDataFmt):
    default_cfg = {
        'backbone_datafmt_cls': 'KTInterDataFmt',
        'num_cluster': 7,
        'centroids_update_epoch': 2,
        'seg_length': 20
    }

    def __init__(self, cfg,train_dict,val_dict,test_dict,feat2type, **kwargs):
        super().__init__(cfg, train_dict, val_dict, test_dict, feat2type, **kwargs)
        self.seg_length = self.default_cfg['seg_length']
        self.add_seg_num_features()
        self.cluster = self.construct_cluster_feats()

    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        self.datafmt_cfg['dt_info']['n_cluster'] = self.default_cfg['num_cluster']


    def __getitem__(self, index):
        dic = super().__getitem__(index)
        # dic['cluster'] = self.cluster[(int(dic['stu_id']), int(dic['seg_seq']))]
        step = len(dic['seg_seq'])
        stu_id = dic['stu_id'].repeat(step)
        # cluster_df = pd.DataFrame([[list(each)] for each in torch.cat((stu_id.unsqueeze(1), dic['seg_seq'].unsqueeze(1)), dim=1).numpy()], columns=['stu_seg_id'])
        # result = pd.merge(cluster_df, self.cluster, on = ['stu_seg_id']).reset_index(drop=True)
        # cluster_id_tensor = torch.Tensor(result['cluster_id'].values)
        # dic['cluster'] = cluster_id_tensor
        dic['cluster'] = np.ones_like(dic['exer_seq'])
        for i in range(step):
            try:
                dic['cluster'][i] = self.cluster.get((int(stu_id[i]), int(dic['seg_seq'][i])))
            except:
                dic['cluster'][i] = 0
        dic['cluster'] = torch.from_numpy(dic['cluster'])
        return dic

    def add_seg_num_features(self):
        self.train_dict['seg_seq'] = self.add_seg_num(self.train_dict)
        self.test_dict['seg_seq'] = self.add_seg_num(self.test_dict)
        if len(self.datafmt_cfg['divide_scale_list']) == 3:
            self.val_dict['seg_seq'] = self.add_seg_num(self.val_dict)

    def add_seg_num(self, data_dict):
        stu_id_in_dict = data_dict['stu_id'].unique()
        seg_seq = torch.zeros_like(data_dict['exer_seq'])
        segs = torch.from_numpy(np.arange(self.datafmt_cfg['window_size']) // self.seg_length)

        for s in stu_id_in_dict:
            mask_s = (data_dict['stu_id'] == s)
            seg_id_s = torch.arange(0, mask_s.sum())

            seg_seq[mask_s] = segs.repeat((len(seg_id_s),1)) + seg_id_s.unsqueeze(dim=1).repeat((1,self.datafmt_cfg['window_size'])) * (self.datafmt_cfg['window_size'] // self.seg_length)

            # seg_seq.append(seg_id_s)
        return seg_seq

    def construct_cluster_feats(self):
        self.stu_count = self.datafmt_cfg['dt_info']['stu_count']
        self.exer_count = self.datafmt_cfg['dt_info']['exer_count']
        self.num_cluster = self.default_cfg['num_cluster']
        self.centroids_update_epoch = self.default_cfg['centroids_update_epoch']
        # self.identifiers = self.default_cfg['identifiers']


        cluster_data = {}
        cluster_data['train_cluster_data'] = self._construct_cluster_matrix(self.train_dict)
        if self.val_dict:
            cluster_data['val_cluster_data'] = self._construct_cluster_matrix(self.val_dict)
        cluster_data['test_cluster_data'] = self._construct_cluster_matrix(self.test_dict)
        cluster = self._k_means_cluster(cluster_data)
        return cluster

    def _construct_cluster_matrix(self, data_dict):
        cluster_data = []
        xtotal = torch.zeros((self.stu_count, self.exer_count))
        x1 = torch.zeros((self.stu_count, self.exer_count))
        x0 = torch.zeros((self.stu_count, self.exer_count))

        for i, s in enumerate(data_dict['stu_id']):
            exer_s = data_dict['exer_seq'][i]
            label_s = data_dict['label_seq'][i]
            seg_id_s = data_dict['seg_seq'][i]

            for si in range(len(seg_id_s.unique())):
                stu_id = s
                mask_seg = (seg_id_s == si)
                # seg_id = seg_id_s[mask_seg]
                problem_ids = exer_s[mask_seg]
                correctness = label_s[mask_seg]

                xtotal[stu_id, problem_ids] += 1
                x1[stu_id, problem_ids] += (correctness == 1).float()
                x0[stu_id, problem_ids] += (correctness != 1).float()

                xsr = x1[stu_id] / (xtotal[stu_id] + 0.0000001)
                xfr = x0[stu_id] / (xtotal[stu_id] + 0.0000001)

                x = torch.nan_to_num(xsr) - torch.nan_to_num(xfr)
                x = torch.cat((x, torch.tensor([stu_id])))
                x = torch.cat((x, torch.tensor([si])))
                cluster_data.append(x.cpu().numpy())
        return cluster_data

    def _euclidean_distance(self, x, y):
        return torch.dist(torch.tensor(x), y, p=2)

    def _k_means_cluster(self, cluster_data):
        cluster = {}
        points = np.array(cluster_data['train_cluster_data'])[:,:-2]

        centroids = torch.from_numpy(np.random.permutation(points)[:self.num_cluster, :])
        points_e = torch.from_numpy(np.expand_dims(points, axis=0))
        centroids_e = torch.from_numpy(np.expand_dims(centroids, axis=1))
        distances = torch.sum((points_e - centroids_e) ** 2, dim=-1)
        indices = torch.argmin(distances, dim=0)
        clusters = [points[np.where(indices == i)[0]] for i in range(self.num_cluster)]
        new_centroids = torch.cat(
            [torch.mean(torch.from_numpy(clusters[i]), dim=0, keepdim=True) for i in range(self.num_cluster)], dim=0)

        # update centroids
        centroids = new_centroids

        for j in range(self.centroids_update_epoch):
            points_e = torch.from_numpy(np.expand_dims(points, axis=0))
            centroids_e = torch.from_numpy(np.expand_dims(centroids, axis=1))
            distances = torch.sum((points_e - centroids_e) ** 2, dim=-1)
            indices = torch.argmin(distances, dim=0)
            clusters = [points[np.where(indices == i)[0]] for i in range(self.num_cluster)]
            new_centroids = torch.cat(
                [torch.mean(torch.from_numpy(clusters[i]), dim=0, keepdim=True) for i in range(self.num_cluster)],
                dim=0)
            # update centroids
            centroids = new_centroids.numpy()

        # cluster for training data and testing data
        for students in cluster_data.values():
            for i in students:
                inst = i[:-2]
                min_dist = float('inf')
                closest_clust = None
                for j in range(self.num_cluster):
                    cur_dist = np.linalg.norm(inst - centroids[j], ord=2)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = j
                # cluster.append([[int(i[-2]), int(i[-1])], closest_clust])
                cluster[int(i[-2]), int(i[-1])] = closest_clust
        return cluster
        # return pd.DataFrame(cluster, columns=['stu_seg_id','cluster_id'])

