import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, utils
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
import PIL

import os
import os.path


def partition_data(args, dataset, datadir, logdir, partition, n_parties, beta=0.4):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if dataset == "femnist":
        X_train, y_train, u_train, X_test, y_test, u_test = load_femnist_data(datadir)
        n_train = y_train.shape[0]
    else:
        raise RuntimeError('This function  can be only uesed on FEMNIST.')

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif partition == "real" and dataset == "femnist":
        num_user = u_train.shape[0]
        user = np.zeros(num_user+1,dtype=np.int32)
        for i in range(1,num_user+1):
            user[i] = user[i-1] + u_train[i-1]
        no = np.random.permutation(num_user)
        batch_idxs = np.array_split(no, n_parties)
        net_dataidx_map = {i:np.zeros(0,dtype=np.int32) for i in range(n_parties)}
        for i in range(n_parties):
            for j in batch_idxs[i]:
                net_dataidx_map[i]=np.append(net_dataidx_map[i], np.arange(user[j], user[j+1]))
                
    elif partition == "transfer-from-femnist":
        stat = np.load("data/femnist/FEMNIST/femnist-dis.npy")
        n_total = stat.shape[0]
        chosen = np.random.permutation(n_total)[:n_parties]
        stat = stat[chosen,:]
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = stat[:,k]
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
  

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

   
            
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

        
def load_femnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = FEMNIST(datadir, train=True, transform=transform, download=False)
    mnist_test_ds = FEMNIST(datadir, train=False, transform=transform, download=False)

    X_train, y_train, u_train = mnist_train_ds.data, mnist_train_ds.targets, mnist_train_ds.users_index
    X_test, y_test, u_test = mnist_test_ds.data, mnist_test_ds.targets, mnist_test_ds.users_index

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    u_train = np.array(u_train)
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    u_test = np.array(u_test)

    return (X_train, y_train, u_train, X_test, y_test, u_test)

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp


    return net_cls_counts


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]
    
    train_file = 'training.pt'
    test_file = 'test.pt'
    _check_exists = False

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train
        if download:
            self.download()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        if  self._check_exists:
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil, os
        # if self._check_exists():
        #     return 

        mkdirs(self.raw_folder)
        mkdirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        try:
            shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
            shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)
        except OSError:
            pass

