from ..common import BaseMid2Cache
import numpy as np
import torch


class M2C_DKTDSC_OP(BaseMid2Cache):
    default_cfg = {
        'n_cluster': 7,
        'centroids_update_epoch': 2,
        'seg_length': 20
    }

    def process(self, **kwargs):
        dt_info = kwargs['dt_info']
        dt_info['n_cluster'] = self.m2c_cfg['n_cluster']
        self.seg_length = self.m2c_cfg['seg_length']
        self.stu_count = dt_info['stu_count']
        self.exer_count = dt_info['exer_count']
        self.num_cluster = self.m2c_cfg['n_cluster']
        self.window_size = dt_info['real_window_size']

        df_train_folds = kwargs['df_train_folds']
        df_valid_folds = kwargs['df_valid_folds']
        df_test_folds = kwargs['df_test_folds']

        # self.dt_info = kwargs['dt_info']
        # self.dt_info['n_pcount_list'] = []
        # self.dt_info['n_rgap_list'] = []
        # self.dt_info['n_sgap_list'] = []

        kwargs['cluster_list'] = []

        for idx, (train_dict, test_dict) in enumerate(zip(df_train_folds, df_test_folds)):
            self.train_dict = train_dict
            self.test_dict = test_dict
            if df_valid_folds is not None and len(df_valid_folds) > 0:
                self.val_dict = df_valid_folds[idx]

            self.add_seg_num_features()
            self.cluster = self.construct_cluster_feats()
            kwargs['cluster_list'].append(self.cluster)

        return kwargs

    def add_seg_num_features(self):
        self.train_dict['seg_seq:token_seq'] = self.add_seg_num(self.train_dict)
        self.test_dict['seg_seq:token_seq'] = self.add_seg_num(self.test_dict)
        if self.val_dict is not None:
            self.val_dict['seg_seq:token_seq'] = self.add_seg_num(self.val_dict)

    def add_seg_num(self, data_dict):
        stu_id_in_dict = np.unique(data_dict['stu_id:token'])
        seg_seq = torch.zeros_like(torch.from_numpy(data_dict['exer_seq:token_seq']))
        segs = torch.from_numpy(np.arange(self.window_size) // self.seg_length)

        for s in stu_id_in_dict:
            mask_s = (data_dict['stu_id:token'] == s)
            seg_id_s = torch.arange(0, mask_s.sum())

            seg_seq[mask_s] = segs.repeat((len(seg_id_s),1)) + seg_id_s.unsqueeze(dim=1).repeat((1,self.window_size)) * (self.window_size // self.seg_length)

            # seg_seq.append(seg_id_s)
        return seg_seq

    def construct_cluster_feats(self):

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

        for i, s in enumerate(data_dict['stu_id:token']):
            exer_s = torch.from_numpy(data_dict['exer_seq:token_seq'][i])
            label_s = torch.from_numpy(data_dict['label_seq:float_seq'][i])
            seg_id_s =  data_dict['seg_seq:token_seq'][i]

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

