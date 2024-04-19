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

    # Initialize the original SAM model and processor
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    print(args)

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
        dataset = SA1BDataset('SA1B', processor=processor, do_crop=args.crop)

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


    print(f'trainloader : {len(train_dataloader)}')
    print(f'testloader : {len(test_dataloader)}')

    # Load the model configuration
    saved_supermodel = SamModel.from_pretrained("logs/2024-04-18--17:05:20.238154_dataset[sa1b]_trainable[em]_epochs[100]_lr[1e-05]_local_bs[24]")

    ofm = OFM(saved_supermodel)

    miou = test(ofm.model,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    print(f'Supermodel from file {count_parameters(saved_supermodel)}: mIoU = {miou}%')

    submodel, params, config = ofm.smallest_model()
    miou = test(submodel,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    print(f'Smallest from file {count_parameters(submodel)} params: mIoU = {miou}%')

    submodel, params, config = ofm.random_resource_aware_model()
    miou = test(submodel,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    print(f'Medium from file {count_parameters(submodel)} params: mIoU = {miou}%')