
from re import L
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
import torch
from dsets import TrafficLightDataset
from util.logconf import logging
import sys
import argparse
import datetime
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.optim as optim
import os
import torch.nn.functional as F
from torchvision import transforms
#from models.FasterRCNN import TrafficLightNetFasterRCNN
import torch.nn as nn
from util.util import enumerateWithEstimate
from testing_functions import non_maximum_suppression, calculate_iou, evaluate_model, plot_dataset_item
import numpy as np
import copy
import shutil
import hashlib
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)



METRICS_TR_LOSS=0
METRICS_PRED_BOXES=1
METRICS_PRED_SCORES=2
METRICS_PRED_LABELS=3
METRICS_GT_BOXES=4
METRICS_GT_LABELS=5
METRICS_SIZE=6

train_transforms= transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    ])
val_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])




def collate_fn(batch):
    return tuple(zip(*batch))

base_path = r"c:\users\alex\.cache\kagglehub\datasets\wjybuqi\traffic-light-detection-dataset\versions\4"
train_dataset=os.path.join(base_path, "train_dataset")


test_dataset=os.path.join(base_path, "test_dataset")

class TrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--learning-rate',
                            help='Learning rate for optimizer',
            default=1e-3,
            type=float,
        )
        parser.add_argument('--num-workers',
                help='Number of worker processes for background data loading',
                default=1,
                type=int,
                         )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=1,
            type=int,
        )
        parser.add_argument('--num-epochs',
            help='Number of epochs to train for',
            default=10,
            type=int,
        )
        parser.add_argument('--tb-prefix',
            default='Traffic Light Detection',
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )
        parser.add_argument('--comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='dwlpt',
        )
        


        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        

        self.use_cuda = torch.cuda.is_available()

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.init_model()

        
        self.target_boxes=[]
        self.target_labels=[]
        

        
     
       
        
        
        self.optimizer, self.lr_scheduler = self.init_optimizer()

        self.totalTrainingSamples_count=0

        self.bbox_loss = nn.SmoothL1Loss()
        self.head_loss = nn.CrossEntropyLoss()
        #self.head_loss = nn.functional.cross_entropy


    def init_optimizer(self):

        optimizer= SGD(self.model.parameters(), lr=self.cli_args.learning_rate, momentum=0.9,weight_decay=5e-4)
        lr_scheduler= StepLR(optimizer, step_size=10, gamma=0.1)

        return optimizer, lr_scheduler


    def init_model(self):
        #model = TrafficLightNetFasterRCNN
        #faster_model=TrafficLightNetFasterRCNN()
        model=fasterrcnn_resnet50_fpn(pretrained=True)

        in_features=model.roi_heads.box_predictor.cls_score.in_features
        num_classes=4
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        

        if torch.cuda.device_count() > 1:
           
            model = torch.nn.DataParallel(model)
           
            
            #faster_model = torch.nn.DataParallel(faster_model)

        model= model.to(self.device)
        #faster_model= faster_model.to(self.device)

        return model

    def init_DL(self):
       dataset= TrafficLightDataset(dataset_path=base_path)

       train_size= int(0.8 * len(dataset))
       val_size= len(dataset) - train_size

       train_subset, val_subset= random_split(dataset, [train_size, val_size])
        
       
       batch_size = self.cli_args.batch_size
       if self.use_cuda:
            batch_size *= torch.cuda.device_count()

       train_dataset= copy.deepcopy(train_subset)
       val_dataset= copy.deepcopy(val_subset)
       train_dataset.dataset = copy.deepcopy(train_subset.dataset)
       val_dataset.dataset = copy.deepcopy(val_subset.dataset)

       train_dataset.dataset.transform = train_transforms

       val_dataset.dataset.transform = val_transforms
       val_dataset.dataset.mode = 'val'

       train_loader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.cli_args.num_workers, collate_fn=collate_fn, pin_memory=self.use_cuda, drop_last=False)
       val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.cli_args.num_workers, collate_fn=collate_fn, pin_memory=self.use_cuda, drop_last=False)
       
       return train_loader, val_loader
    
    def init_test_DL(self):

        test_dataset= TrafficLightDataset(dataset_path=base_path, mode="test")
          
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
                batch_size *= torch.cuda.device_count()
    
         
         
        test_loader= DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=self.cli_args.num_workers, collate_fn=collate_fn, pin_memory=self.use_cuda, drop_last=False)
         
        return test_loader
   
    # def train_epoch(self):
    #    self.model.train()
    #    total_loss = 0.0
    #    scaler = torch.cuda.amp.GradScaler()
    #    for batch_tup in self.training_dataset:  # DataLoader batches
    #     images, bboxes, colors = batch_tup

    #     # images = torch.stack(images).to(self.device, non_blocking=True)
    #     # bboxes = torch.stack(bboxes).to(self.device, non_blocking=True)
    #     # colors = torch.stack(colors).to(self.device, non_blocking=True)
         
        
    # # Move images & targets to device
    #     images=images.to(self.device, non_blocking=True)
    #     target_bboxes=bboxes.to(self.device, non_blocking=True)
    #     target_colors=colors.to(self.device, non_blocking=True)

       
    

        
    #     with torch.cuda.amp.autocast():

    #     # Forward pass returns a dict of losses in training mode
    #        pred_bboxes, pred_colors = self.model(images)

    #        loss = self.bbox_loss(pred_bboxes.float(), target_bboxes.float()) + \
    #             self.color_loss(pred_colors.float(), target_colors.long())
        
        
    #     #self.optimizer.step()
        
    #     scaler.scale(loss).backward()
    #     scaler.step(self.optimizer)
    #     scaler.update()
    #     self.optimizer.zero_grad()
    #     total_loss += loss
           
    #     self.totalTrainingSamples_count = self.totalTrainingSamples_count + len(self.training_dataset)

    #    avg_loss = total_loss / len(self.training_dataset)
    #    return avg_loss

    def doTrain(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                start_ndx=0,  # Training starts at index 0 in the metrics tensor
                mode='train'
                
                
            )

            loss_var.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            num_imgs= len(val_dl.dataset)
            valMetrics_g = {
              METRICS_PRED_BOXES: [None] * num_imgs,
               METRICS_PRED_SCORES: [None] * num_imgs,
               METRICS_PRED_LABELS: [None] * num_imgs,
               METRICS_GT_BOXES: [None] * num_imgs,
              METRICS_GT_LABELS: [None] * num_imgs,
                                        }
            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                current_batch= batch_ndx * val_dl.batch_size
                loss_var = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                    start_ndx=current_batch,  # Validation starts at index 0 in the metrics tensor
                    mode='val'
                    
                    
                )
        return valMetrics_g

    def doTest(self, epoch_ndx, test_dl):
        with torch.no_grad():
            self.model.eval()
            num_imgs= len(test_dl.dataset)
            testMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(test_dl.dataset),
                device=self.device,
            )


            batch_iter = enumerateWithEstimate(
                test_dl,
                "E{} Testing".format(epoch_ndx),
                start_ndx=self.cli_args.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                start= batch_ndx * test_dl.batch_size
                predictions = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    test_dl.batch_size,
                    testMetrics_g,
                    start_ndx=start,
                    mode='test'
                    
                    
                )
        return testMetrics_g.to('cpu')

    
        

        
    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, start_ndx, mode): 
        
        test_loss=0.0
        
        
        

        if mode == 'train':
           start_ndx = batch_ndx * batch_size
           end_ndx = start_ndx + len(batch_tup[0])
           images, targets, image_name=batch_tup

           

           images = [img.to(self.device, non_blocking=True) for img in images]

           

           for t in targets:
               t['boxes']=t['boxes'].to(self.device, non_blocking=True)
               t['labels']=t['labels'].to(self.device, non_blocking=True)
               t['area']=t['area'].to(self.device, non_blocking=True)
               t['iscrowd']=t['iscrowd'].to(self.device, non_blocking=True)
               t['image_id']=t['image_id'].to(self.device, non_blocking=True) 

           

           loss_dict= self.model(images, targets)

          
          
           train_loss = sum(loss for loss in loss_dict.values())
           

           
           metrics_g[METRICS_TR_LOSS, start_ndx:end_ndx] = train_loss.detach()
           return train_loss.mean()

        elif mode == 'test':
           
            images, image_name=batch_tup
            images = [img.to(self.device, non_blocking=True) for img in images]
            

            predictions= self.model(images)
            return predictions

        elif mode =='val':
            start_ndx = batch_ndx * batch_size
            end_ndx = start_ndx + len(batch_tup[0])
            images, targets, image_name=batch_tup
            images = [img.to(self.device, non_blocking=True) for img in images]
            for t in targets:
               t['boxes']=t['boxes'].to(self.device, non_blocking=True)
               t['labels']=t['labels'].to(self.device, non_blocking=True)
               t['area']=t['area'].to(self.device, non_blocking=True)
               t['iscrowd']=t['iscrowd'].to(self.device, non_blocking=True)
               t['image_id']=t['image_id'].to(self.device, non_blocking=True) 

            predictions= self.model(images, targets)

            if len(predictions) != len(targets):
                log.warning(f"Batch {batch_ndx}: Number of predictions ({len(predictions)}) does not match number of targets ({len(targets)}). Skipping batch.")
                return None
            
            
            for i, pred in enumerate(predictions):
                pred_boxes = pred['boxes'].detach().cpu()
                pred_scores = pred['scores'].detach().cpu()
                pred_labels = pred['labels'].detach().cpu()
                gt_boxes = targets[i]['boxes'].detach().cpu()
                gt_labels = targets[i]['labels'].detach().cpu()
               
                metrics_g[METRICS_PRED_BOXES][start_ndx + i] = pred_boxes
                metrics_g[METRICS_PRED_SCORES][start_ndx + i] = pred_scores
                metrics_g[METRICS_PRED_LABELS][start_ndx + i] = pred_labels

                metrics_g[METRICS_GT_BOXES][start_ndx + i] = gt_boxes
                metrics_g[METRICS_GT_LABELS][start_ndx + i] = gt_labels






            
            
           

            




            
            


            
                
        

        
    def logMetrics(self, epoch_ndx, mode_str, metrics_t, score_threshold=0.5):
      

      if mode_str == "train":
            avg_loss = metrics_t[METRICS_TR_LOSS].mean().item()
            log.info(f"Epoch {epoch_ndx}, {mode_str} Loss: {avg_loss:.4f}")
            
      elif mode_str == "val":
            classes = [1,2,3]
            APs=[]
            iou_threshold = 0.5
           
                
            tp_total_global = 0
            fp_total_global = 0
            gt_total_global = 0
            for class_id in classes:
                all_class_preds = [] 
                
                total_gt_class = 0  
                for img_idx in range(len(metrics_t[METRICS_PRED_BOXES])):
                    pred_boxes = metrics_t[METRICS_PRED_BOXES][img_idx]
                    pred_scores = metrics_t[METRICS_PRED_SCORES][img_idx]
                    pred_labels = metrics_t[METRICS_PRED_LABELS][img_idx]
                    gt_boxes = metrics_t[METRICS_GT_BOXES][img_idx]
                    gt_labels = metrics_t[METRICS_GT_LABELS][img_idx]


                    if pred_boxes is None or gt_boxes is None:
                        continue

                    score_mask = pred_scores >= score_threshold

                    pred_boxes = pred_boxes[score_mask]
                    pred_scores = pred_scores[score_mask]
                    pred_labels= pred_labels[score_mask]



                    pred_mask = pred_labels == class_id
                    gt_mask = gt_labels == class_id
                    
                    p_boxes_c= pred_boxes[pred_mask]
                    p_scores_c = pred_scores[pred_mask]
                    g_boxes_c = gt_boxes[gt_mask]
                    
                    

                    total_gt_class += len(g_boxes_c)
                    if len(p_boxes_c) == 0:
                        continue
                    if len(g_boxes_c) > 0:
                        ious = calculate_iou(p_boxes_c, g_boxes_c)
                        matched_gt = set()


                        _, sorted_indices = torch.sort(p_scores_c, descending=True)
                        
                        for p_idx in sorted_indices:
                             p_idx = p_idx.item()
                             curr_ious= ious[p_idx]
                             best_iou, best_gt_idx = torch.max(curr_ious, dim=0)
                             best_gt_idx = best_gt_idx.item()
                 
                            

                             if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                                  all_class_preds.append([p_scores_c[p_idx].item(), 1])
                                  matched_gt.add(best_gt_idx)
                                  tp_total_global += 1
                             else:
                                  all_class_preds.append([p_scores_c[p_idx].item(), 0])
                                  fp_total_global += 1
                    else:
                          for score in p_scores_c:
                              all_class_preds.append([score.item(), 0])
                              fp_total_global += 1

                    gt_total_global += total_gt_class

                if total_gt_class == 0:
                    continue 

                if not all_class_preds:
                    APs.append(0.0)
                    continue

                all_class_preds.sort(key=lambda x: x[0], reverse=True)

                tp_list= np.array([x[1] for x in all_class_preds])
                fp_list = 1 - tp_list
                tp_cumsum = np.cumsum(tp_list)
                fp_cumsum = np.cumsum(fp_list)

                recalls = tp_cumsum / total_gt_class
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                

                

                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))

                for i in range(len(precisions) - 1, 0, -1):
                    precisions[i - 1] = max(precisions[i - 1], precisions[i])

                ap = np.trapz(precisions, recalls)
                APs.append(ap)
                

            total_gt_objects = sum([len(labels) if labels is not None else 0 for labels in metrics_t[METRICS_GT_LABELS]])
            fn_total_global = total_gt_objects - tp_total_global
            precision_global = tp_total_global / (tp_total_global + fp_total_global + 1e-8)
            recall_global = tp_total_global / (tp_total_global + fn_total_global + 1e-8)
            mAP = np.mean(APs) if APs else 0.0
            f1_score= 2 * (precision_global * recall_global) / (precision_global + recall_global + 1e-8)

            metrics_dict={}
            metrics_dict['precision_global']=precision_global
            metrics_dict['recall_global']=recall_global
            metrics_dict['f1_score']=f1_score
            metrics_dict['mAP']=mAP
            log.info("Epoch {}, mode {}, Precision {precision_global:.4f}, Recall {recall_global:.4f}, F1 Score {f1_score:.4f}, mAP {mAP:.4f}"
                     .format(
                         epoch_ndx,
                         mode_str,
                         **metrics_dict)
                     )

            return mAP


                
    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        best_score = 0.0
        train_dl, val_dl=self.init_DL()
        test_dl=self.init_test_DL()
        for epoch in range(self.cli_args.num_epochs):
            log.info("Epoch {} of {}, {}/{} testing of size {} batches of size {}*{}".format(
                epoch,
                self.cli_args.num_epochs,
                len(train_dl),
                len(val_dl),
                len(test_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            trnMetrics_g = self.doTrain(epoch, train_dl)
            self.logMetrics(epoch, "train", trnMetrics_g)
            valMetrics_g = self.doValidation(epoch, val_dl)
            score=self.logMetrics(epoch, "val", valMetrics_g)
            
            best_score = max(score, best_score)
            self.saveModel("model", epoch, score == best_score)
    
    def saveModel(self, type_str, epoch_ndx, isBest= False):
        # model = self.segmentation_model

        

        base_dir = r"C:\Users\Alex\source\repos\Stair Detection Kaggle Dataset\Stair Detection Kaggle Dataset"
        file_path=os.path.join(
            base_dir,
            'data-unversioned',
            'part2',
            'models',
             self.cli_args.tb_prefix, 
             '{}_{}_{}.{}.state'.format(
                    type_str,
                   self.time_str,  
                    self.cli_args.comment,    
                    self.totalTrainingSamples_count,
                ),
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        


        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("")  # or save model state later
                log.info("Created new file at {}".format(file_path))
        else:
            log.info("File already exists at {}".format(file_path))


        model = self.model
        if isinstance(model, torch.nn.DataParallel):
             model = model.module

        
        state = {
                'sys_argv': sys.argv,
                'time': str(datetime.datetime.now()),
                'epoch_ndx': epoch_ndx,
                'model_state': model.state_dict(),
                'model_name': type(model).__name__,
                'epoch': epoch_ndx,
                'optimizer_state': self.optimizer.state_dict(),
            }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                base_dir,
               'data-unversioned',
                'part2',
                'models',
                self.cli_args.tb_prefix,
                f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')

            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

                                    




if __name__ == "__main__":
    app = TrainingApp()
    app.main()