import argparse
import os
from dataset import get_loader
from solver import Solver
import time


def get_test_info(config):
    if config.sal_mode == 'NJU2K':
        image_root = 'dataset/test/NJU2K_test/'
        image_source = 'dataset/test/NJU2K_test/test.lst'
    elif config.sal_mode == 'STERE':
        image_root = 'dataset/test/STERE/'
        image_source = 'dataset/test/STERE/test.lst'
    elif config.sal_mode == 'RGBD135':
        image_root = 'dataset/test/RGBD135/'
        image_source = 'dataset/test/RGBD135/test.lst'
    elif config.sal_mode == 'LFSD':
        image_root = 'dataset/test/LFSD/'
        image_source = 'dataset/test/LFSD/test.lst'
    elif config.sal_mode == 'NLPR':
        image_source = 'dataset/test/NLPR/test.lst'
    elif config.sal_mode == 'SIP':
        image_root = 'dataset/test/SIP/'
        image_source = 'dataset/test/SIP/test.lst'
    else:
        raise Exception('Invalid config.sal_mode')

    config.test_root = image_root
    config.test_list = image_source


def main(config):
    if config.mode == 'train':
        #train_loader = get_loader(config)
        pp=4
        # if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
        #     os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        # config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        # train = Solver(train_loader, None, config)        
        # train.train()  
        
        
    elif config.mode == 'test':
        get_test_info(config)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_folder): os.makedirs(config.test_folder)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")
      

if __name__ == '__main__':
    resnet_path = 'pretrained/resnet101-5d3b4d8f.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)  #   default=0.00005    Learning rate resnet:5e-5    0.0004
    #    parser.add_argument('--lr', type=float, default=0.00005)  # 0.001
    
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay  default=0.0005     0.0009
    #parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--momentum', type=float, default=0.99)

    #parser.add_argument('--wd', type=float, default=0.001)  # Weight decay
    #parser.add_a/home/rabia/Desktoprgument('--momentum', type=float, default=0.99)
    #parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=320)

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')
    #parser.add_argument('--device_id', type=str, default='cuda:0,1')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)  # pretrained abackbone model
    parser.add_argument('--epoch', type=int, default=50)    # parser.add_argument('--epoch', type=int, default=45)

    #parser.add_argument('--epoch', type=int, default=45)
    #parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now

    #parser.add_argument('--batch_size', type=int, default=10)  # only support 1 now
    #parser.add_argument('--num_thread', type=int, default=1)
    
    parser.add_argument('--batch_size', type=int, default = 1)  # only support 1 now
    parser.add_argument('--num_thread', type=int, default=4)
    
    parser.add_argument('--load', type=str, default='')  # pretrained JL-DCF model
    parser.add_argument('--save_folder', type=str, default='checkpoints/')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='./dataset/train')
    #parser.add_argument('--train_root', type=str, default='D:/work/python/RGBDcollection')
    #parser.add_argument('--train_list', type=str, default='D:/work/python/RGBDcollection/train.lst')
    parser.add_argument('--train_list', type=str, default='./dataset/train/train.lst')

    # Testing settings
    #checkpoints/demo-08
    #parser.add_argument('--model', type=str, default='checkpoints/demo-xx/epoch_xx.pth')  # Snapshot
    parser.add_argument('--model', type=str, default='checkpoints/demo-09/epoch_50.pth')  # Snapshot
    #parser.add_argument('--test_folder', type=str, default='test/demoxx/xx/STERE/')  # Test results saving folder
    parser.add_argument('--test_folder', type=str, default='test/demoo/STERE/')  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='LFSD',
                        choices=['NJU2K', 'NLPR', 'STERE', 'RGBD135', 'LFSD', 'SIP'])  # Test image dataset
    #parser.add_argument('--sal_mode', type=str, default='STERE',
    # Misc
    #parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    config = parser.parse_args()


    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    get_test_info(config)

    main(config)



# # args.device_id = torch.cuda.
#     if not os.path.exists(config.save_folder):
#         os.mkdir(config.save_folder)

#     get_test_info(config)

#     main(config)