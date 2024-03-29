import os
from transformers import SamModel, SamConfig, SamProcessor
from raffm import RaFFM
from torch.utils.data import DataLoader, Subset
from utility import *
from dataset import SA1BDataset, MitoDataset
from torch.optim import Adam
import monai
from statistics import mean

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #torch.cuda.empty_cache()
    args = get_args()

    # Initialize the original SAM model and processor
    original_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # RaFFM configuration and submodel initialization
    elastic_config = {
        "atten_out_space": [768],
        "inter_hidden_space": [1024],
        "residual_hidden_space": [1024],
    }
    raffm_model = RaFFM(original_model, elastic_config=elastic_config)
    submodel, params, config = raffm_model.random_resource_aware_model()
    submodel = submodel.to(device)  # Move submodel to GPU

    print(f'Original model size : {count_parameters(original_model)}')
    print(f'Submodel model size : {count_parameters(submodel)}')


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
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False)
        
        else:
            subset_dataset = Subset(dataset, indices=range(0,10000,1))
            train_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        
        if args.test_subset:
            subset_dataset = Subset(dataset, indices=range(args.train_subset,args.train_subset + args.test_subset,1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        else:
            subset_dataset = Subset(dataset, indices=range(10000,len(dataset),1))
            test_dataloader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)


    print(f'trainloader : {len(train_dataloader)}')
    print(f'testloader : {len(test_dataloader)}')

    # Calculate IoUs and average IoU before training

    # miou = test(original_model,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    # print(f'original_model pre-train mIoU : {miou}')
    
    miou = test(submodel,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    print(f'submodel pre-train mIoU : {miou}')

    
    # make sure we only compute gradients for mask decoder
    for name, param in submodel.named_parameters():
      if name.startswith("prompt_encoder"): # or name.startswith("mask_decoder") or #name.startswith("vision_encoder")
        param.requires_grad_(False)


    # Initialize the optimizer and the loss function
    optimizer = Adam(list(submodel.vision_encoder.parameters()) + list(submodel.mask_decoder.parameters()), lr=1e-8, weight_decay=0)

    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    #Training loop
    submodel.to(device)

    submodel.train()
    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader, disable=args.no_verbose):
            # forward pass
            outputs = submodel(pixel_values=batch["pixel_values"].to(device),
                            input_boxes=batch["input_boxes"].to(device),
                            multimask_output=False)
            
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks, batch["original_sizes"], batch["reshaped_input_sizes"], binarize=False,
            )

            # compute loss
            predicted_masks = masks[0]  # outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            
            # Calculate focal loss and dice loss
            loss_focal = focal_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            loss_dice = dice_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
            
            # Combine the losses
            loss = 20 * loss_focal + loss_dice

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

    # Save the model's state dictionary to a file
    torch.save(submodel.state_dict(), "models/sa1b_submodel_checkpoint.pth")
    torch.save(config,"models/sa1b_submodel_config.pth")

    """**Inference**"""

    # Load the model configuration
    config = torch.load("models/sa1b_submodel_config.pth")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # Create an instance of the model architecture with the loaded configuration
    my_saved_submodel, total_params = raffm_model.resource_aware_model(config)
    #Update the model by loading the weights from saved file.
    my_saved_submodel.load_state_dict(torch.load("models/sa1b_submodel_checkpoint.pth"))

    my_saved_submodel.to(device)

    # #Training loop
    # submodel.to(device)

    # submodel.train()
    # for epoch in range(args.epochs):
    #     epoch_losses = []
    #     for batch in tqdm(train_dataloader, disable=args.no_verbose):
    #         # forward pass
    #         outputs = submodel(pixel_values=batch["pixel_values"].to(device),
    #                         input_boxes=batch["input_boxes"].to(device),
    #                         multimask_output=False)
            
    #         masks = processor.image_processor.post_process_masks(
    #             outputs.pred_masks, batch["original_sizes"], batch["reshaped_input_sizes"], binarize=False,
    #         )

    #         # compute loss
    #         predicted_masks = masks[0] #outputs.pred_masks.squeeze(1)
    #         ground_truth_masks = batch["ground_truth_mask"].float().to(device)
    #         loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

    #         # backward pass (compute gradients of parameters w.r.t. loss)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         # optimize
    #         optimizer.step()
    #         epoch_losses.append(loss.item())

    #     print(f'EPOCH: {epoch}')
    #     print(f'Mean loss: {mean(epoch_losses)}')

    # # Save the model's state dictionary to a file
    # torch.save(submodel.state_dict(), "models/sa1b_submodel_checkpoint.pth")
    # torch.save(config,"models/sa1b_submodel_config.pth")

    # """**Inference**"""

    # # Load the model configuration
    # config = torch.load("models/sa1b_submodel_config.pth")
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # # Create an instance of the model architecture with the loaded configuration
    # my_saved_submodel, total_params = raffm_model.resource_aware_model(config)
    # #Update the model by loading the weights from saved file.
    # my_saved_submodel.load_state_dict(torch.load("models/sa1b_submodel_checkpoint.pth"))

    # my_saved_submodel.to(device)


    #Testing

    miou = test(my_saved_submodel,test_dataloader,batch_size=args.batch_size,disable_verbose=args.no_verbose,processor=processor)
    print(f'submodel post-train mIoU : {miou}')

