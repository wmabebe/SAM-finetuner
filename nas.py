import os
import datetime
from transformers import SamModel, SamConfig, SamProcessor
from ofm import OFM
from torch.utils.data import DataLoader, Subset
from utility import *
from logger import init_logs, get_logger
from dataset import SA1BDataset, MitoDataset
import timeit

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()

    NOW = str(datetime.datetime.now()).replace(" ","--")
    log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_dir = './logs/{}_dataset[{}]_trainable[{}]_epochs[{}]_lr[{}]_local_bs[{}]/'. \
        format(NOW,args.dataset, args.trainable, args.epochs, args.lr, args.batch_size)
    

    init_logs(log_file_name, args, log_dir)
    args.logger = get_logger()

    # Initialize the original SAM model and processor
    original_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # OFM configuration and submodel initialization
    elastic_config = {
        "atten_out_space": [256,512,768],
        "inter_hidden_space": [512, 768, 1024],
        "residual_hidden_space": [512, 768, 1024],
    }
    ofm = OFM(original_model, elastic_config=elastic_config)
    args.supermodel = ofm

    args.logger.info(f'Original model size : {count_parameters(args.supermodel.model)} params')

    if args.dataset == 'mito':

        # Create an instance of the SAMDataset
        train_dataset = load_dataset("datasets/mitochondria/training.tif", "datasets/mitochondria/training_groundtruth.tif")
        test_dataset = load_dataset("datasets/mitochondria/testing.tif", "datasets/mitochondria/testing_groundtruth.tif")
        train_dataset = MitoDataset(dataset=train_dataset, processor=processor)
        test_dataset = MitoDataset(dataset=test_dataset, processor=processor)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(train_dataset, indices=range(args.train_subset))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        
        # Apply subset for shorter testing
        if args.test_subset:
            subset_dataset = Subset(test_dataset, indices=range(args.test_subset))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    
    elif args.dataset == 'sa1b':
        dataset = SA1BDataset('SA1B', processor=processor)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = custom_collate_fn) #collate_fn = custom_collate_fn
        
        else:
            subset_dataset = Subset(dataset, indices=range(0,10000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn = custom_collate_fn)
        
        if args.test_subset:
            subset_dataset = Subset(dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = custom_collate_fn) #collate_fn = custom_collate_fn
        else:
            subset_dataset = Subset(dataset, indices=range(10000,len(dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn = custom_collate_fn)


    args.logger.info(f'trainloader : {len(train_dataloader)}')
    args.logger.info(f'testloader : {len(test_dataloader)}')

    # Calculate IoUs and average IoU before training

    miou = test(args.supermodel.model,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    args.logger.info(f'supermodel pre-NAS mIoU : {miou}%')

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader
    args.processor = processor
    args.device = device
    args.log_dir = log_dir
    
    args.logger.info(f'NAS Training starts')
    start = timeit.default_timer()        
    train_nas(args)
    end = timeit.default_timer()
    args.logger.info(f'NAS Training ends : {round(end-start,4)} seconds')    


    # Load the model configuration
    model_config = SamConfig.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the model architecture with the loaded configuration
    saved_supermodel = SamModel(config=model_config)
    #Update the model by loading the weights from saved file.
    saved_supermodel.load_state_dict(torch.load(f"{log_dir}{args.dataset}_supermodel_checkpoint.pth"))

    # set the device to cuda if available, otherwise use cpu
    saved_supermodel.to(device)

    miou = test(saved_supermodel,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    args.logger.info(f'supermodel post-NAS mIoU : {miou}%')