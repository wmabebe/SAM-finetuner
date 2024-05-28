from transformers import SamModel, SamConfig, SamProcessor
from ofm import OFM
from torch.utils.data import DataLoader, Subset
from utility import *
from dataset import SA1BDataset, MitoDataset
from torch.optim import Adam

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()

    print(args)

    # Initialize the original SAM model and processor
    original_model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # OFM configuration and submodel initialization
    elastic_config = {
        "atten_out_space": [768],
        "inter_hidden_space": [1024],
        "residual_hidden_space": [1024],
    }
    ofm_model = OFM(original_model, elastic_config=elastic_config)
    submodel, params, config = ofm_model.smallest_model()


    print(f'Original model size : {count_parameters(original_model)} params : {count_parameters(original_model)}')
    print(f'Submodel model size : {count_parameters(submodel)} params : {count_parameters(submodel)}')


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
        dataset = SA1BDataset('dataset/SA1B', processor=processor)

        # Apply subset for shorter training
        if args.train_subset:
            subset_dataset = Subset(dataset, indices=range(0,args.train_subset,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = sa1b_collate_fn) #collate_fn = custom_collate_fn
        
        else:
            subset_dataset = Subset(dataset, indices=range(0,10000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn = sa1b_collate_fn)
        
        if args.test_subset:
            subset_dataset = Subset(dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn = sa1b_collate_fn) #collate_fn = custom_collate_fn
        else:
            subset_dataset = Subset(dataset, indices=range(10000,len(dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn = sa1b_collate_fn)


    print(f'trainloader : {len(train_dataloader)}')
    print(f'testloader : {len(test_dataloader)}')

    # Calculate IoUs and average IoU before training

    miou = test(original_model,test_dataloader,device=device,disable_verbose=args.no_verbose,processor=processor)
    print(f'original_model pre-train mIoU : {miou}')
    
    miou = test(submodel,test_dataloader,device=device,disable_verbose=args.no_verbose,processor=processor)
    print(f'submodel pre-train mIoU : {miou}')

    
    # Freeze modules that are not set as trainable
    if 'p' not in args.trainable:
        for name, param in submodel.named_parameters():
            if name.startswith("prompt_encoder"):
                param.requires_grad_(False)
    if 'e' not in args.trainable:
         for name, param in submodel.named_parameters():
            if name.startswith("vision_encoder"):
                param.requires_grad_(False)
    if 'm' not in args.trainable:
         for name, param in submodel.named_parameters():
            if name.startswith("mask_decoder"):
                param.requires_grad_(False)


    #Select the parameters to train
    trainable_params = []
    if 'e' in args.trainable:
        trainable_params += list(submodel.vision_encoder.parameters())
    if 'm' in args.trainable:
        trainable_params += list(submodel.mask_decoder.parameters())
    if 'p' in args.trainable:
        trainable_params += list(submodel.prompt_encoder.parameters())
    
    # Initialize the optimizer and the loss function
    optimizer = Adam(trainable_params, lr=1e-5, weight_decay=0)

    train(submodel, optimizer, args.epochs, train_dataloader, processor, device, no_verbose=args.no_verbose)

    
    # Save the model's state dictionary to a file
    torch.save(submodel.state_dict(), "models/sa1b_submodel_checkpoint.pth")
    torch.save(config,"models/sa1b_submodel_config.pth")

    #args.supermodel.save_ckpt(f'{args.log_dir}')

    """Inference"""

    # Load the model configuration
    config = torch.load("models/sa1b_submodel_config.pth")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the model architecture with the loaded configuration
    my_saved_submodel, total_params = ofm_model.resource_aware_model(config)
    #Update the model by loading the weights from saved file.
    my_saved_submodel.load_state_dict(torch.load("models/sa1b_submodel_checkpoint.pth"))



    miou = test(my_saved_submodel,test_dataloader,device=device, disable_verbose=args.no_verbose,processor=processor)
    print(f'submodel post-train mIoU : {miou}')

