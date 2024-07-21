import matplotlib.pyplot as plt
import numpy as np
import torch
import config
import cv2

def yolo2pixel(bbox):
    '''
    Transforms yolo coordinates of the box to pixel coordinates. 
    
    Arguments:
        - bbox: yolo coordinates [xc, yc, width, height]
    
    Returns: 
        - pixel coordinates list [xmin, xmax, ymin, ymax]
    '''
    xc = bbox[0]
    yc = bbox[1]
    width = bbox[2]
    height = bbox[3]
      
    xmin = xc - (width/2)          
    xmax = xc + (width/2)         
    ymin = yc - (height/2)            
    ymax = yc + (height/2)
        
    nbox = [xmin, ymin, xmax, ymax]
    
    return nbox

def iou(
    boxes_preds, boxes_labels, 
    box_format="midpoint",
    epsilon=1e-6
):
    """
    Calculates intersection over union for bounding boxes.
    
    :param boxes_preds (tensor): Bounding box predictions of shape (BATCH_SIZE, 4)
    :param boxes_labels (tensor): Ground truth bounding box of shape (BATCH_SIZE, 4)
    :param box_format (str): midpoint/corners, if boxes (x,y,w,h) format or (x1,y1,x2,y2) format
    :param epsilon: Small value to prevent division by zero.
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] 
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = (box1_area + box2_area - intersection + epsilon)

    iou = intersection / union
    #print(f'IOU is numpy: {iou.numpy()}')

    return iou

def non_max_supression(bboxes, 
                       iou_threshold=config.IOU_THRESHOLD, 
                       score_threshold=config.SCORE_THRESHOLD, 
                       box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [x1, y1, x2, y2, confidence, class_id] MY FORMAT VERSION       
        iou_threshold (float): threshold where predicted bboxes is correct
        score_threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[4] > score_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[5] != chosen_box[5]
            or iou(
                torch.tensor(chosen_box[:4]),
                torch.tensor(box[:4]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


# ______________________________________________________________ #
# ____________________      Pred Boxes      ____________________ #
# ______________________________________________________________ #
def get_best_box_and_class(out):
    
    conf_1 = out[..., 4:5]
    conf_2 = out[..., 9:10]
    confs = torch.cat((conf_1, conf_2), dim=-1)
    _, idx = torch.max(confs, keepdim=True, dim=-1)
    
    best_boxes = idx*out[..., 5:10] + (1-idx)*out[..., 0:5]
    
    _, class_idx = torch.max(out[..., 10:12], keepdim=True, dim=-1)
    
    best_out = torch.cat((best_boxes, class_idx), dim=-1)

    return best_out

def get_bboxes_from_model_out(model_out, 
                              iou_threshold=config.IOU_THRESHOLD, 
                              score_threshold=config.SCORE_THRESHOLD):

    model_out = get_best_box_and_class(model_out.detach().to('cpu'))
    
    c2b_mtx = np.zeros((config.S, config.S, 2))
    for j in range(config.S):
        for i in range(config.S):
            c2b_mtx[i, j, 0] = j
            c2b_mtx[i, j, 1] = i

    model_out = model_out.numpy()
    out_xy = model_out[..., :2]
    out_rest = model_out[..., 2:]

    c2b_xy = (c2b_mtx+out_xy)/config.S
    out = np.concatenate((c2b_xy, out_rest), axis=-1)
    #print(f'Concat out\n {out}')

    bboxes_flat = np.reshape(out, (config.S*config.S, 5+1)) # Replace 5+C by 5+1, as we filtered best class before (get_best_box_and_class)
    bboxes_list = [bbox for bbox in bboxes_flat.tolist()]

    nms_pred_bboxes = non_max_supression(
        bboxes_list,
        iou_threshold=iou_threshold, 
        score_threshold=score_threshold, 
        box_format="midpoint")

    return nms_pred_bboxes


# ______________________________________________________________ #
# ____________________      True Boxes      ____________________ #
# ______________________________________________________________ #
def get_bboxes_from_label_mtx(label_mtx):
    '''
    Receives a label_mtx, as yielded by dataset or dataloader and returns a list of bounding boxes.
    
    Arguments:
        - label_mtx
    
    Returns:
        - bboxes_list: list with all cells containing score = 1
            [xcell, ycell, w, h, score, smoke, fire] -> [x, y, w, h, 1, smoke, fire]
    '''

    c2b_mtx = np.zeros((config.S, config.S, 2))
    for j in range(config.S):
        for i in range(config.S):
            c2b_mtx[i, j, 0] = j
            c2b_mtx[i, j, 1] = i

    label_mtx = label_mtx.numpy()
    label_xy = label_mtx[..., :2]
    label_rest = label_mtx[..., 2:]

    c2b_xy = (c2b_mtx+label_xy)/config.S
    out = np.concatenate((c2b_xy, label_rest), axis=-1)
    #print(f'Concat out\n {out}')

    bboxes_list = np.reshape(out, (config.S*config.S, 5+config.C))

    bboxes_list = [bbox for bbox in bboxes_list.tolist() if bbox[4]==1]

    return bboxes_list


# ______________________________________________________________ #
# ____________________        Plots         ____________________ #
# ____________________ True & Pred Boxes    ____________________ #
# ______________________________________________________________ #
def plot_grid(img):
    '''
    Plot grid on top of the picture
    '''
      
    cell_size = int(config.IMG_W / config.S)
    
    # Draw horizontal lines
    for i in range(1, config.S):
        cv2.line(img, (0, cell_size*i), (config.IMG_W-1, cell_size*i), config.GRID_COLOR, 1)
    # Draw vertical lines
    for j in range(1, config.S):
        cv2.line(img, (cell_size*j, 0), (cell_size*j, config.IMG_H-1), config.GRID_COLOR, 1)
        
    return img

    
def plot_dataset_img(img, label_mtx, grid=False):
    '''
    It draws the bounding boxes over the image.

    Arguments:
        - ori_img: original image with no modification or letterbox
        - label_mtx: [xcell, ycell, w, h, score=1, smoke, fire], tensor (7, 7, 12)
        - grid: plot grid over the image

    Returns:
        - pic: picture with bounding boxes on top of original picture
    '''

    # NEVER remove copy() or use np.ascontiguousarray()
    # Convert pytorch tensor to numpy
    img = img.permute(1, 2, 0) * 256
    img = img.numpy().astype(np.uint8).copy()   
       
    if grid == True:
        img = plot_grid(img)
    
    bboxes = get_bboxes_from_label_mtx(label_mtx)

    for i,(xc, yc, w, h, score, smoke, fire) in enumerate(bboxes):
        xmin, ymin, xmax, ymax = xc - w/2, yc - h/2, xc + w/2, yc + h/2
        box = np.array([xmin, ymin, xmax, ymax]).astype(np.float32)
        box[0] = box[0]*config.IMG_W
        np.clip(box[0], 0, None)
        box[1] = box[1]*config.IMG_H
        np.clip(box[1], 0, None)
        box[2] = box[2]*config.IMG_W - 1 # avoid out of limits due to rounding
        box[3] = box[3]*config.IMG_H - 1 # avoid out of limits due to rounding
        box = box.round().astype(np.int32).tolist()
        #print(f'Box after conversion\n{box}')
        if smoke == 1:
            class_id = 0
        elif fire == 1:
            class_id = 1
        else:
            print("Wrong Class ID")
        name = config.CLASSES[class_id]
        color = config.BBOX_COLORS[name]
        cv2.rectangle(img, box[:2], box[2:], color, 1) 
        if box[1] < 30:
            if class_id == 0:
                cv2.rectangle(img, [box[0], box[1]+15], [box[0]+55, box[1]], color, -1) 
            else:
                cv2.rectangle(img, [box[0], box[1]+15], [box[0]+25, box[1]], color, -1) 
            cv2.putText(img,name,(box[0], box[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0],
                        thickness=1)  # 0.5 -> font size
        else:
            if class_id == 0:
                cv2.rectangle(img, [box[0], box[1]-20], [box[0]+55, box[1]], color, -1) 
            else:
                cv2.rectangle(img, [box[0], box[1]-20], [box[0]+25, box[1]], color, -1) 
            cv2.putText(img,name,(box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0],
                        thickness=1)  # 0.5 -> font size

    return img


def plot_predicted_img(img, model_out, grid=False):
    '''
    It draws the bounding boxes over the image.

    Arguments:
        - ori_img: original image with no modification or letterbox
        - model_out: [xcell, ycell, w, h, score, smoke, fire], tensor (7, 7, 12)
        - grid: plot grid over the image

    Returns:
        - pic: picture with bounding boxes on top of original picture
    '''

    # NEVER remove copy() or use np.ascontiguousarray()
    # Convert pytorch tensor to numpy
    img = img.permute(1, 2, 0) * 256
    img = img.numpy().astype(np.uint8).copy()   
       
    if grid == True:
        img = plot_grid(img)
    
    bboxes = get_bboxes_from_model_out(model_out)

    for xc, yc, w, h, score, class_id in bboxes:
        xmin, ymin, xmax, ymax = xc - w/2, yc - h/2, xc + w/2, yc + h/2
        box = np.array([xmin, ymin, xmax, ymax]).astype(np.float32)
        box[0] = box[0]*config.IMG_W
        np.clip(box[0], 0, None)
        box[1] = box[1]*config.IMG_H
        np.clip(box[1], 0, None)
        box[2] = box[2]*config.IMG_W - 1 # avoid out of limits due to rounding
        box[3] = box[3]*config.IMG_H - 1 # avoid out of limits due to rounding
        box = box.round().astype(np.int32).tolist()
        
        class_id = int(class_id)
        name = config.CLASSES[class_id]
        color = config.BBOX_COLORS[name]
        name += str(f' {score:.3f}')
        
        cv2.rectangle(img, box[:2], box[2:], color, 1) 
        if box[1] < 30:
            if class_id == 0:
                cv2.rectangle(img, [box[0], box[1]+15], [box[0]+105, box[1]], color, -1) 
            else:
                cv2.rectangle(img, [box[0], box[1]+15], [box[0]+80, box[1]], color, -1) 
            cv2.putText(img,name,(box[0], box[1] + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0],
                        thickness=1)  # 0.5 -> font size
        else:
            if class_id == 0:
                cv2.rectangle(img, [box[0], box[1]-20], [box[0]+105, box[1]], color, -1) 
            else:
                cv2.rectangle(img, [box[0], box[1]-20], [box[0]+80, box[1]], color, -1) 
            cv2.putText(img,name,(box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 0],
                        thickness=1)  # 0.5 -> font size

    return img


# ______________________________________________________________ #
# ____________________         Plot         ____________________ #
# ____________________      N IMAGES        ____________________ #
# _________________ [ori1, pred1, ori2, pred2] _________________ #
# ______________________________________________________________ #
def plot_n_images(loader, model, n_imgs, save_name):
    '''
    Plots 4 pictures in each row: [ori1, pred1, ori2, pred2]
    '''
    
    n_imgs = int(n_imgs / 2) * 2 # Make n_imgs an even number
    rows = int(n_imgs/2)
    cols = 4
    plot_img_height = n_imgs
    
    fig, ax = plt.subplots(rows, cols, figsize=(10, n_imgs)) # 32 pics -> (10, 40) / 64 
    
    img_idx = 0
    
    for (img, label) in loader:
        
        half_batch = int(config.BATCH_SIZE/2)
        for i in range(half_batch): # index for Half Batch_Size
            
            #print(f'Batch Index: {i}')
            ax_idx = int(img_idx/4)
            #print(f'AX idx: {ax_idx}')
            #print(f'Img idx before update: {img_idx}')
    
            # Left Half: Original, Predicted
            plt.subplot(rows, cols, img_idx+1)
            if img_idx == 0:
                ax[ax_idx][0].set_title("Original") # Set title for colum 0

            ori_pic_1 = plot_dataset_img(img[2*i], label[2*i], grid=False)
            ax[ax_idx][0].imshow(ori_pic_1)
            ax[ax_idx][0].set_axis_off()

            plt.subplot(rows, cols, img_idx+2)
            if img_idx == 0:
                ax[ax_idx][1].set_title("Predicted") # Set title for colum 1

            img_to_model_1 = img[2*i].unsqueeze(dim=0).to(config.DEVICE)
            pred_out_1 = model(img_to_model_1)
            pred_out_1 = pred_out_1.permute(0, 2, 3, 1)
            pred_pic_1 = plot_predicted_img(img[2*i], pred_out_1, grid=False)
            ax[ax_idx][1].imshow(pred_pic_1)
            ax[ax_idx][1].set_axis_off()

            # Right Half: Original, Predicted
            plt.subplot(rows, cols, img_idx+3)
            if img_idx == 0:
                ax[ax_idx][2].set_title("Original") # Set title for colum 2

            ori_pic_2 = plot_dataset_img(img[2*i+1], label[2*i+1], grid=False)
            ax[ax_idx][2].imshow(ori_pic_2)
            ax[ax_idx][2].set_axis_off()

            plt.subplot(rows, cols, img_idx+4)
            if img_idx == 0:
                ax[ax_idx][3].set_title("Predicted") # Set title for colum 3

            img_to_model_2 = img[2*i+1].unsqueeze(dim=0).to(config.DEVICE)
            pred_out_2 = model(img_to_model_2)
            pred_out_2 = pred_out_2.permute(0, 2, 3, 1)
            pred_pic_2 = plot_predicted_img(img[2*i+1], pred_out_2, grid=False)
            ax[ax_idx][3].imshow(pred_pic_2)
            ax[ax_idx][3].set_axis_off()

            img_idx += 4 # Move to next row
            #print(f'Img idx after update: {img_idx}')
            
            if int(img_idx/2) == n_imgs:
                #print("Break Inner Loop")
                break
        
        if int(img_idx/2) == n_imgs:
            #print("Break Outer Loop: data loader")
            break
    
    plt.tight_layout(pad=0.5)

    if save_name is not None:
        plt.savefig(save_name + '.png')
        plt.close()
    else:
        plt.show()    



class LogMetrics():
    
    def __init__(self):
        
        self.smoke_AP = []
        self.smoke_AR = []
        self.fire_AP = []
        self.fire_AR = []
        self.mAP = []
        
        self.smoke_metrics_dic = {
            'AP': self.smoke_AP,
            'AR': self.smoke_AR,
        }
        self.fire_metrics_dic = {
            'AP': self.fire_AP,
            'AR': self.fire_AR,
        }
        self.mean_metrics_dic = {'mAP': self.mAP}
                
    def update_metrics(self, metrics):
        for k, v in metrics.items():
            if k == 'AP' or k == 'AR':
                self.smoke_metrics_dic[k].append(v[0])
                self.fire_metrics_dic[k].append(v[1])
            elif k == 'mAP':
                self.mean_metrics_dic[k].append(v.item())
            else:
                print("Wrong Metric Passed")
            
    def get_metrics(self):
        return {
            'Smoke': self.smoke_metrics_dic, 
            'Fire': self.fire_metrics_dic, 
            'mAP': self.mean_metrics_dic
        }
    
class LogLosses():

    def __init__(self):
        self.total = []
        self.box = []
        self.conf = []
        self.noobj = []
        self.classif = []
        self.losses = {
            'Total': self.total,
            'Box': self.box,
            'Conf': self.conf,
            'No Obj': self.noobj,
            'Class': self.classif,
        }
                
    def update_losses(self, metrics):
        for k, v in metrics.items():
            self.losses[k].append(v)
            
    def get_losses(self):
        return self.losses

class LogLR():

    def __init__(self, log_path):
        self.log_path = log_path
        self.lr = []
                
    def log_lr(self, lr):
        self.lr.append(lr)
            
    def plot_lr(self, epochs):
        plt.plot(epochs, self.lr)
        plt.title("Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.savefig(self.log_path + "learning_rate.png")
        plt.close()    
    

class PlotMetrics():
    def __init__(self, log_path, model_name, loss_or_metric):
        self.log_path = log_path
        self.model_name = model_name
        self.loss_or_metric = loss_or_metric
    
    def __plot_one_metric__(
        self,
        smoke_or_fire: str, 
        metric_name: str, 
        train_metric: list, 
        val_metric: list, 
        epochs: list):
        '''
        Plot one metric including Train and Val data
        '''
        # Fix names and position to save files and plot titles
        if metric_name == 'Loss':
            save_name = f'_{metric_name}__{smoke_or_fire}'
        else:
            if metric_name == 'mAP':
                metric_name = 'mAP:0.50'
                smoke_or_fire = ''
                save_name = f'_Metric__{metric_name}' 
            else:
                save_name = f'_Metric__{smoke_or_fire}_{metric_name}' 
            
        plt.plot(epochs, train_metric, label = "Train" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.plot(epochs, val_metric, label = "Val" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.title("Train vs Val:" + f' {smoke_or_fire}' + f' {metric_name}')
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        if metric_name != 'Loss':
            plt.ylim([0,1])
        plt.legend()
        plt.grid()       
        plt.savefig(self.log_path + f'{self.model_name}_' + save_name + ".png")
        plt.close()    

    def plot_all_metrics(self, train_metrics, val_metrics, epochs):
        
        # Plot Losses
        if self.loss_or_metric == "Loss":
            for (key_train, value_train), (key_val, value_val) in zip(train_metrics.items(), val_metrics.items()):
                self.__plot_one_metric__(
                    smoke_or_fire = key_train, 
                    metric_name = 'Loss', 
                    train_metric = value_train, 
                    val_metric = value_val, 
                    epochs = epochs)
            self.__plot_losses_together__(train_metrics, 'Train', epochs)
            self.__plot_losses_together__(val_metrics, 'Val', epochs)

        elif self.loss_or_metric == "Metric":
            for k_train, k_val in zip(train_metrics.keys(), val_metrics.keys()):
                for (key_train, value_train), (key_val, value_val) in zip(train_metrics[k_train].items(), val_metrics[k_val].items()):
                    self.__plot_one_metric__(
                        smoke_or_fire = k_train, 
                        metric_name = key_train, 
                        train_metric = value_train, 
                        val_metric = value_val, 
                        epochs = epochs)
                    
        else:
            raise SystemExit("Wrong Loss or Metric request to Plot")
            
    def __plot_losses_together__(self, losses, train_or_val, epochs):
        '''
        Plot all train or val losses in the same figure
        '''
        for k in losses.keys():
            plt.plot(epochs, losses[k], label = f'{k}' + ' Loss')
        plt.title(f'{train_or_val}' + ' Loss')
        plt.xlabel("Epochs")
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig(self.log_path + f'{self.model_name}_' + f'_Losses__{train_or_val}' + ".png")
        plt.close()    
    
def print_metrics_to_logger(train_or_val, losses, metrics, mAP_available=False):
    str_metric = ('Total Loss'.ljust(11) + '|Box Loss'.ljust(11) + '|Conf Loss'.ljust(11) + 
                 '|NoObj Loss'.ljust(11) + '|Class Loss'.ljust(11))
    str_hyp = ('-----------'.ljust(11) + '|----------'.ljust(11) + '|----------'.ljust(11) + 
                '-----------'.ljust(11) + '-----------'.ljust(11))
    str_value = (f'{losses["Total"]:.2f}'.ljust(11) + 
                 f'|{losses["Box"]:.2f}'.ljust(11) + 
                 f'|{losses["Conf"]:.2f}'.ljust(11) +
                 f'|{losses["No Obj"]:.2f}'.ljust(11) +
                 f'|{losses["Class"]:.2f}'.ljust(11))

    if mAP_available == True:
        str_metric += ('|    |' + 'Smoke AP'.ljust(11) + '|Smoke AR'.ljust(11) + 
                      '|Fire AP'.ljust(11) + '|Fire AR'.ljust(11) + '|mAP:0.50'.ljust(11))
        str_hyp += ('|    |' + '-----------'.ljust(11) + '|----------'.ljust(11) + '|----------'.ljust(11) + 
                '-----------'.ljust(11) + '-----------'.ljust(11))
        str_value += ('|    |' + f'{metrics["AP"][0]:.4f}'.ljust(11) + 
                      f'|{metrics["AR"][0]:.4f}'.ljust(11) + 
                      f'|{metrics["AP"][1]:.4f}'.ljust(11) +
                      f'|{metrics["AR"][1]:.4f}'.ljust(11) +
                      f'|{metrics["mAP"]:.4f}'.ljust(11))
    else:
        str_metric += '|     | No mAP in this epoch '

    return f'{train_or_val}\n{str_metric}|\n{str_hyp}|\n{str_value}|\n'

def save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, 
    checkpoint_name) 
    
def load_checkpoint(model_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Loading Model. Trained during {start_epoch} epochs")
    return start_epoch
    
    
    
    