import sys
from tqdm import tqdm
import config
import modules.utils as utils
import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

map_metric = MeanAveragePrecision(
    box_format='xyxy',
    iou_thresholds=[config.IOU_THRESHOLD],
    class_metrics=True, # Enables separated metrics for each class
    #average='micro',
    extended_summary=False).to(config.DEVICE)

# ______________________________________________________________ #
# ____________________      True Boxes      ____________________ #
# ______________________________________________________________ #
def get_true_boxes(target_out):
  
    c2b_mtx = np.zeros((config.S, config.S, 2))
    for j in range(config.S):
        for i in range(config.S):
            c2b_mtx[i, j, 0] = j
            c2b_mtx[i, j, 1] = i

    target_out = target_out.numpy()
    out_xy = target_out[..., :2]
    out_rest = target_out[..., 2:]

    c2b_xy = (c2b_mtx+out_xy)/config.S
    out = np.concatenate((c2b_xy, out_rest), axis=-1)

    bboxes_flat = np.reshape(out, (config.S*config.S, 5+2)) # [x, y, w, h, 1 or 0, smoke, fire]
    bboxes_list = [bbox for bbox in bboxes_flat.tolist() if bbox[4]==1]

    xyxy = []
    labels = []
    for ground_truth in bboxes_list:
        x, y, w, h, _, smoke, fire = ground_truth
        xmin, ymin, xmax, ymax = x - w/2, y - h/2, x + w/2, y + h/2
        box = np.array([xmin, ymin, xmax, ymax]).astype(np.float32)
        box[0] = box[0] * config.IMG_W
        box[1] = box[1] * config.IMG_H
        box[2] = box[2] * config.IMG_W
        box[3] = box[3] * config.IMG_H
        xyxy.append(box)
        if smoke == 1:
            labels.append(int(0))
        elif fire == 1:
            labels.append(int(1)) 
        else:
            raise ValueError("Wrong Class")

    xyxy = torch.tensor(np.array(xyxy), dtype=torch.float32).to(config.DEVICE) 
    labels = torch.tensor(np.array(labels), dtype=torch.int32).to(config.DEVICE) 
    
    return [{"boxes": xyxy,
             "labels": labels}]


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

def get_pred_boxes(model_out, iou_threshold=config.NMS_IOU_THRESHOLD, score_threshold=config.SCORE_THRESHOLD):

    model_out = get_best_box_and_class(model_out)
    
    c2b_mtx = np.zeros((config.S, config.S, 2))
    for j in range(config.S):
        for i in range(config.S):
            c2b_mtx[i, j, 0] = j
            c2b_mtx[i, j, 1] = i

    model_out = model_out.numpy()
    
    # Bound all numbers to [0, 1] after removing sigmoid and softmax
    model_out = np.clip(a=model_out, a_min=0, a_max=1)
    
    out_xy = model_out[..., :2]
    out_rest = model_out[..., 2:]

    c2b_xy = (c2b_mtx+out_xy)/config.S
    out = np.concatenate((c2b_xy, out_rest), axis=-1)
    #print(f'Concat out\n {out}')

    bboxes_flat = np.reshape(out, (config.S*config.S, 5+1)) # Replace 5+C by 5+1, as we filtered best class before (get_best_box_and_class)
    bboxes_list = [bbox for bbox in bboxes_flat.tolist()]

    nms_pred = utils.non_max_supression(bboxes_list,
                                        iou_threshold=iou_threshold, 
                                        score_threshold=score_threshold, 
                                        box_format="midpoint")
    xyxy = []
    scores = []
    labels = []
    for pred in nms_pred:
        x, y, w, h, score, class_id = pred
        xmin, ymin, xmax, ymax = x - w/2, y - h/2, x + w/2, y + h/2
        box = np.array([xmin, ymin, xmax, ymax]).astype(np.float32)
        box[0] = box[0] * config.IMG_W
        box[1] = box[1] * config.IMG_H
        box[2] = box[2] * config.IMG_W
        box[3] = box[3] * config.IMG_H
        xyxy.append(box)
        scores.append(score)
        labels.append(int(class_id))

    xyxy = torch.tensor(np.array(xyxy), dtype=torch.float32).to(config.DEVICE) 
    scores = torch.tensor(np.array(scores), dtype=torch.float32).to(config.DEVICE) 
    labels = torch.tensor(np.array(labels), dtype=torch.int32).to(config.DEVICE) 
    
    return [{"boxes": xyxy,
             "scores": scores,
             "labels": labels}]

# ______________________________________________________________ #
# ____________________     mAP Function     ____________________ #
# ______________________________________________________________ #
def torchmetrics_mAP(loader, model, metric=map_metric, device=config.DEVICE):

    metric.reset()
    model.eval()
    
    loop = tqdm(loader, desc='Validating', leave=True)
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        
        # Remove Permute from the model
        out = out.permute(0, 2, 3, 1)

        # Mean Average Precision
        for idx in range(x.shape[0]):
            target_boxes = get_true_boxes(y[idx].detach().to('cpu'))
            pred_boxes = get_pred_boxes(out[idx].detach().to('cpu'))
            metric.update(preds = pred_boxes, target = target_boxes) 

    meanAP = metric.compute()
    metric.reset()
    #print(f'Val mAP = {meanAP["map_50"]:.4f}')
    
    return (
        {'mAP': meanAP['map_50'],
         'AP': [meanAP['map_per_class'][0].item(), meanAP['map_per_class'][1].item()],
         'AR': [meanAP['mar_100_per_class'][0].item(), meanAP['mar_100_per_class'][1].item()]
        }
    )