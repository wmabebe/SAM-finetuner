import torch
import torch.nn as nn
from pycocotools import mask as mask_utils
from tqdm import tqdm
from utility import compute_iou, get_trainable_parameters, get_optimizer_and_scheduler
import timeit
import copy
from monai.losses import DiceLoss, DiceFocalLoss
import random


class SA1B_NAS_Trainer:
    #Initialize dataloader, args
    def __init__(self,args):
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.best_smallest_submodel_iou = 0
    
    def eval(self,model,map=None):
        mIoU = []
        model.eval()
        model = nn.DataParallel(model).to(self.device)
        for idx,(inputs, images, labels, boxes, points) in enumerate(tqdm(self.test_dataloader, disable=self.no_verbose)): 

            #Set num_objects to 64 or num of points in the data with fewest points
            num_objects = min(64, min([len(pts) for pts in points]))

            # Filter num_objects prompt indices from points and boxes
            input_boxes, input_points = [], []
            for i,d in enumerate(inputs):
                input_boxes.append(d['input_boxes'][:,:num_objects, :])
                input_points.append(d['input_points'][:num_objects, :, :])
                labels[i] = labels[i][:num_objects]
            input_boxes = torch.stack(input_boxes)
            input_points = torch.stack(input_points)
                

            data = {'pixel_values': torch.stack([d['pixel_values'] for i,d in enumerate(inputs)]),
                 'original_sizes' : torch.stack([d['original_sizes'] for i,d in enumerate(inputs)]),
                 'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for i,d in enumerate(inputs)]),
                 'input_boxes' : input_boxes, 
                 'input_points' : input_points }
            
            data["pixel_values"] = data["pixel_values"].squeeze(1)
            data['original_sizes'] = data['original_sizes'].squeeze(1)
            data['reshaped_input_sizes'] = data['reshaped_input_sizes'].squeeze(1)
            data['input_points'] = data['input_points'].squeeze(1)
            data['input_boxes'] = data['input_boxes'].squeeze(1)
            

            with torch.no_grad():
                if self.test_prompt == 'p':
                    outputs = model(pixel_values=data["pixel_values"].to(self.device),
                                    input_points=data["input_points"].to(self.device),
                                    multimask_output=True)
                elif self.test_prompt == 'b': 
                    outputs = model(pixel_values=data["pixel_values"].to(self.device),
                                    input_boxes=data["input_boxes"].to(self.device),
                                    multimask_output=True)
                
                masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), 
                                                                        data["original_sizes"].cpu(), 
                                                                        data["reshaped_input_sizes"].cpu())
                scores = outputs.iou_scores

                print(f'masks : {len(masks)}')

                #loop through batch (images)
                for i, mask in enumerate(masks):
                    pts, bxs = points[i], boxes[i]
                    #loop through objects
                    for j,(pred_1, pred_2, pred_3) in enumerate(mask):
                        gt = mask_utils.decode(labels[i][j]['segmentation'])
                        gt = torch.from_numpy(gt)
                        _, mious = compute_iou([pred_1, pred_2, pred_3],[gt,gt,gt])

                        mIoU.append(max(mious))

                        #Record ground truth with prediction
                        if map != None:
                            pt, bx = [pts[j]], bxs[j]
                            map[f'img-{i}-obj-{j}-pred-0'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_1,mious[0])
                            map[f'img-{i}-obj-{j}-pred-1'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_2,mious[1])
                            map[f'img-{i}-obj-{j}-pred-2'] = (images[i],torch.tensor(pt),torch.tensor(bx),gt,pred_3,mious[2])


        model = model.module
        model = model.to('cpu')
        mIoU = torch.tensor(mIoU)
        average_mIoU = torch.mean(mIoU).item()
        return average_mIoU, mIoU, map
    
    def training_step(self, supermodel, model, data, labels, optimizer, scheduler, processor, device, ragged_tensors, soft_labels=None, prompt='pb',loss_func='dice'):
    
        #seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        if loss_func == 'dice':
            loss_func = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        elif loss_func == 'dicefocal':
            loss_func = DiceFocalLoss(lambda_focal=0.95,lambda_dice=0.05,sigmoid=True, squared_pred=True, reduction='mean')
            
        local_grad = {k: v.cpu() for k, v in model.state_dict().items()}

        model.to(device)
        model = nn.DataParallel(model)

        model.train()
        optimizer.zero_grad()

        #forward pass randomly selects either box or point prompts for mask generation
        if prompt == 'pob':
            if random.choice(['point', 'box']) == 'point':
                outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_points=data["input_points"].to(device),
                            multimask_output=False)
            else:
                outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_boxes=data["input_boxes"].to(device),
                            multimask_output=False)
        #forward with point and box
        elif prompt == 'p+b':
            outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_points=data["input_points"].to(device),
                            input_boxes=data["input_boxes"].to(device),
                            multimask_output=False)
        #forward with point only
        elif prompt == 'p':
            outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_points=data["input_points"].to(device),
                            multimask_output=False)
        #forward with box only
        elif prompt == 'b':
            outputs = model(pixel_values=data["pixel_values"].to(device),
                            input_boxes=data["input_boxes"].to(device),
                            multimask_output=False)

        #post process outputs and retreive masks    
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks, data["original_sizes"], data["reshaped_input_sizes"], binarize=False
        )

        print(f'masks : {masks.shape if type(masks) == torch.tensor else len(masks)}')
        print(f'gts : {labels.shape if type(labels) == torch.tensor else len(labels)}')


        loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        for idx, mask in enumerate(masks):
            outs = mask.squeeze(1).to(self.device)
            gts, preds = labels[idx], []
            for gt in gts:
                gt = mask_utils.decode(gt['segmentation'])
                gt = torch.from_numpy(gt)
                preds.append(gt)
            preds = torch.stack(preds).to(self.device)
            loss += loss_func(preds,outs)


        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()
        # loss.sum().backward()
        optimizer.step()
        scheduler.step()

        model = model.module
        model = model.to('cpu')

        with torch.no_grad():
            for k, v in model.state_dict().items():
                local_grad[k] = local_grad[k] - v.cpu()

        supermodel.apply_grad(local_grad)

        train_metrics = {
            "train_loss": loss.sum().item(),
            "params": model.config.num_parameters,
        }
        return train_metrics
    
    def single_step_encoder(self,submodel,data, gts, model_size, do_test=False):
        trainable_params = get_trainable_parameters(submodel, 'e')
        optimizer, scheduler = get_optimizer_and_scheduler(trainable_params,self.lr)
        start_train = timeit.default_timer()
        metrics = self.training_step_encoder(self.pretrained, self.supermodel, submodel, data, gts, optimizer, scheduler, self.device)
        end_train = timeit.default_timer()
        if do_test:
            start_test = timeit.default_timer()
            miou, _, _ = self.eval(submodel)
            end_test = timeit.default_timer()
            #Save the best supermodel in a separate folder
            if model_size == 'Smallest' and miou > self.best_smallest_submodel_iou:
                self.supermodel.save_ckpt(f'{self.log_dir}Best/')
                self.best_smallest_submodel_iou = miou
            metrics['test_miou'] = miou
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, test time {round(end_test-start_test,4)}, metrics {metrics}')
        else:
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')
    
    
    def single_step(self,submodel,data,gts,ragged_tensors, model_size, do_test=False):
        trainable_params = get_trainable_parameters(submodel, self.trainable)
        optimizer, self.scheduler = get_optimizer_and_scheduler(trainable_params,self.lr,self.weight_decay, self.scheduler)
        start_train = timeit.default_timer()
        metrics = self.training_step(self.supermodel, submodel, data, gts, optimizer, self.scheduler, 
                                        self.processor, self.device, ragged_tensors, soft_labels=None, 
                                        prompt=self.train_prompt, loss_func=self.loss)
        end_train = timeit.default_timer()
        if do_test:
            start_test = timeit.default_timer()
            miou, _, _ = self.eval(submodel)
            end_test = timeit.default_timer()
            #Save the best supermodel in a separate folder
            if model_size == 'Smallest' and miou > self.best_smallest_submodel_iou:
                self.supermodel.save_ckpt(f'{self.log_dir}Best/')
                self.best_smallest_submodel_iou = miou
            metrics['test_miou'] = miou
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, test time {round(end_test-start_test,4)}, metrics {metrics}')
        else:
            self.logger.info(f'\t{model_size} submodel train time : {round(end_train-start_train,4)}, metrics {metrics}')

    
    def train(self):
        #Training loop
        trainable_params = get_trainable_parameters(self.supermodel.model, self.trainable)
        _, self.scheduler = get_optimizer_and_scheduler(trainable_params,self.lr,self.weight_decay)
        ragged_tensors = False
        for epoch in range(self.epochs):
            self.logger.info(f'EPOCH {epoch}: starts')
            #Train encoder only halfway, then finetune end-2-end
            #self.trainable = 'e' if epoch < self.epochs / 2 else 'em' 
            start_epoch = timeit.default_timer()
            for idx,(inputs, images, labels, boxes, points) in enumerate(tqdm(self.train_dataloader, disable=self.no_verbose)): 

                #Set num_objects to 64 or num of points in the data with fewest points
                num_objects = min(64, min([len(pts) for pts in points]))

                # Filter num_objects prompt indices from points and boxes
                input_boxes, input_points = [], []
                for i,d in enumerate(inputs):
                    indices = random.sample(range(d['input_boxes'].size(1)), num_objects)
                    input_boxes.append(d['input_boxes'][:,indices, :])
                    input_points.append(d['input_points'][indices, :, :])
                    labels[i] = [labels[i][j] for j in indices]
                input_boxes = torch.stack(input_boxes)
                input_points = torch.stack(input_points)
                    

                data = {'pixel_values': torch.stack([d['pixel_values'] for i,d in enumerate(inputs)]),
                    'original_sizes' : torch.stack([d['original_sizes'] for i,d in enumerate(inputs)]),
                    'reshaped_input_sizes' : torch.stack([d['reshaped_input_sizes'] for i,d in enumerate(inputs)]),
                    'input_boxes' : input_boxes, 
                    'input_points' : input_points }
                
                data["pixel_values"] = data["pixel_values"].squeeze(1)
                data['original_sizes'] = data['original_sizes'].squeeze(1)
                data['reshaped_input_sizes'] = data['reshaped_input_sizes'].squeeze(1)
                data['input_points'] = data['input_points'].squeeze(1)
                data['input_boxes'] = data['input_boxes'].squeeze(1)
                
                #set to True to test the submodels after training step
                do_test = (idx == len(self.train_dataloader) - 1)
                
                #Intervals at which to save the model checkpoint
                save_interval = len(self.train_dataloader) // self.save_interval if len(self.train_dataloader) > 10 else 1
                do_save = ((idx + 1) % save_interval == 0)
                
                soft_preds = None
                
                #Train largest submodel (= supernet)
                if 'l' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = (
                            copy.deepcopy(self.supermodel.model),
                            self.supermodel.total_params,
                            {},
                        )
                    
                    if 'e' in self.trainable and 'm' in self.trainable:
                        self.single_step(submodel,data,labels,ragged_tensors,model_size='Largest',do_test=do_test)
                    else:
                        self.single_step_encoder(submodel,data, labels, model_size='Largest', do_test=do_test)

                #Train smallest model
                if 's' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.smallest_model()
                    if 'e' in self.trainable and 'm' in self.trainable:
                        self.single_step(submodel,data,labels,ragged_tensors,model_size='Smallest',do_test=do_test)
                    else:
                        self.single_step_encoder(submodel,data, labels, model_size='Smallest', do_test=do_test)

                #Train arbitrary medium sized model
                if 'm' in self.sandwich:
                    submodel, submodel.config.num_parameters, submodel.config.arch = self.supermodel.random_resource_aware_model()
                    if 'e' in self.trainable and 'm' in self.trainable:
                        self.single_step(submodel,data,labels,ragged_tensors,model_size='Medium',do_test=do_test)
                    else:
                        self.single_step_encoder(submodel,data, labels, model_size='Medium', do_test=do_test)

                # Save the supermodel state dictionary to a file
                if do_save:
                    self.supermodel.save_ckpt(f'{self.log_dir}')
                    self.logger.info(f'\tInterval {(idx + 1) // save_interval}: Model checkpoint saved.')

            end_epoch = timeit.default_timer()

            # Save the supermodel state dictionary to a file
            self.supermodel.save_ckpt(f'{self.log_dir}')
            self.logger.info(f'EPOCH {epoch}: ends {round(end_epoch-start_epoch,4)} seconds. Model checkpoint saved.')   
        