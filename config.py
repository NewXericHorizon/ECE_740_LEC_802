class Config:
    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size
    context_amount = 0.5                   # context amount

    # training related
    num_per_epoch = 1166                   # num of samples per epoch
    train_batch_size = 8                   # training batch size
    train_num_workers = 32                 # number of workers of train dataloader
    lr = 1e-2
    end_lr = 1e-5                          # learning rate of SGD
    momentum = 0.9                         # momentum of SGD
    weight_decay = 5e-4                    # weight decay of optimizator
    epoch = 50                             # total epoch
    seed = 1234                            # seed to sample training videos
    radius = 16                            # radius of positive label
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 3                          # number of scales
    scale_lr = 0.59                        # scale learning rate
    response_up_stride = 16                # response upsample stride
    response_sz = 17                       # response size
    train_response_sz = 15                 # train response size
    window_influence = 0.176               # window influence
    scale_penalty = 0.9745                 # scale penalty
    total_stride = 8                       # total stride of backbone
    sample_type = 'uniform'
    gray_ratio = 0.25
    blur_ratio = 0.15

config = Config()
