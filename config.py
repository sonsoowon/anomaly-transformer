from omegaconf import OmegaConf

train_config = OmegaConf.create({
    # TODO: cuda를 사용하면 자꾸 OOM 에러 발생!
    'use_cuda': False,

    'path': {
        'train': "dataset/MSL/MSL_train.npy",
        'test': "dataset/MSL/MSL_test.npy",
        'test_label': "dataset/MSL/MSL_test_label.npy",
        'save_model': "checkpoint"
    },

    # Train params
    'window_size': 100,
    'd_model': 55,
    'head_num': 5,
    'layers': 3,
    'd_ff': 64,
    'dropout': 0.1,
    'activation': 'gelu',

    # Loss
    'lambda_': 3,

    # Early Stopping
    'patience': 5,
    'verbose': True,
    'delta': 0,

    'batch_size': 32,
    'lr': 0.0001,
    'epoch': 10,

    'anomaly_ratio': 0.001
})
