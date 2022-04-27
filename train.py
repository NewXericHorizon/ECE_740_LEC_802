import sys
import os
from mindspore.common import  set_seed
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.dataset as ds
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
import mindspore.dataset.transforms.py_transforms as py_transforms
from config import config
from alexnet import SiameseAlexNet
from dataset import Pair
from custom_transforms import ToTensor, RandomStretch, RandomCrop, CenterCrop
from got10k.datasets import *

sys.path.append(os.getcwd())



def train():
    """set train """

    set_seed(config.seed)
    random_crop_size = config.instance_size - 2 * config.total_stride
    train_z_transforms = py_transforms.Compose([
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)),
        ToTensor()
    ])
    train_x_transforms = py_transforms.Compose([
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size),
                   config.max_translate),
        ToTensor()
    ])
    
    # create dataset
    seq = GOT10k('dataset', subset='train', return_meta=True)
    train_dataset = Pair(seq, train_z_transforms, train_x_transforms)
    dataset = ds.GeneratorDataset(train_dataset, ["exemplar_img", "instance_img"], shuffle=True,
                                  num_parallel_workers=config.train_num_workers)
    dataset = dataset.batch(batch_size=8, drop_remainder=True)
     
    #set network
    network = SiameseAlexNet(train=True)
    decay_lr = nn.exponential_decay_lr(config.lr,
                                      config.end_lr/config.lr,
                                      total_step=config.epoch * config.num_per_epoch,
                                      step_per_epoch=config.num_per_epoch,
                                      decay_epoch=config.epoch)
    
#     decay_lr = nn.polynomial_decay_lr(config.lr,
#                                       config.end_lr,
#                                       total_step=config.epoch * config.num_per_epoch,
#                                       step_per_epoch=config.num_per_epoch,
#                                       decay_epoch=config.epoch,
#                                       power=1.0)
    
    optim = nn.SGD(params=network.trainable_params(),
                   learning_rate=decay_lr,
                   momentum=config.momentum,
                   weight_decay=config.weight_decay)


    loss_scale_manager = DynamicLossScaleManager()
    model = Model(network,
                  optimizer=optim,
                  loss_scale_manager=loss_scale_manager,
                  metrics=None,
                  amp_level='O3')
    config_ck_train = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=20)
    ckpoint_cb_train = ModelCheckpoint(prefix='SiamFC',
                                       directory='./models/siamfc',
                                       config=config_ck_train)
    time_cb_train = TimeMonitor(data_size=config.num_per_epoch)
    loss_cb_train = LossMonitor()

    model.train(epoch=config.epoch,
                train_dataset=dataset,
                callbacks=[time_cb_train, ckpoint_cb_train, loss_cb_train],
                dataset_sink_mode=True
                )


if __name__ == '__main__':

    device_id = 0
    device_target = "Ascend"
    DEVICENUM = int(os.environ.get("DEVICE_NUM", 1))
    DEVICETARGET = "Ascend"
    if DEVICETARGET == "Ascend":
        context.set_context(
            mode=context.GRAPH_MODE,
            device_id=device_id,
            save_graphs=False,
            device_target=device_target)
        if  DEVICENUM > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=DEVICENUM,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    # train
    train()
