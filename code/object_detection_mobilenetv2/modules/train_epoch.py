from tqdm import tqdm
import modules.metrics as metrics
import config
import torch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_fn(loader, model, optimizer, loss_fn, loss_l1_lambda, metric, device, calculate_mAP = False):
    
    print(f'Learning Rate = {get_lr(optimizer=optimizer)}\n')

    model.train()
    loop = tqdm(loader, desc='Training', leave=True)
    train_mean_loss = []
    mean_box_loss = []
    mean_confidence_loss = []
    mean_noobj_loss = []
    mean_class_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        
        # Remove Permute from the model
        out = out.permute(0, 2, 3, 1)
        
        train_loss = loss_fn(ground_truth=y, 
                             predictions=out)


        # hyperparameter for L1 regularization
        if loss_l1_lambda != 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            train_loss = train_loss + loss_l1_lambda * l1_norm       
        
        # Gradient Descent
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENTS_CLIP_NORM)
        optimizer.step()


        # MSE Loss
        xy_loss, wh_loss, obj_loss, noobj_loss, class_loss = loss_fn.get_last_losses()
        # Appending each loss
        train_mean_loss.append(train_loss.item())
        box_loss = xy_loss + wh_loss
        mean_box_loss.append(box_loss)
        mean_confidence_loss.append(obj_loss)
        mean_noobj_loss.append(noobj_loss)
        mean_class_loss.append(class_loss)
        
        # Mean Average Precision
        if calculate_mAP == True:
            for idx in range(x.shape[0]):
                target_boxes = metrics.get_true_boxes(y[idx].detach().to('cpu'))
                pred_boxes = metrics.get_pred_boxes(out[idx].detach().to('cpu'))
                metric.update(preds = pred_boxes, target = target_boxes)

    train_mean_loss_out = sum(train_mean_loss)/len(train_mean_loss)
    #print("\nTRAIN losses")
    mean_box_loss_out = sum(mean_box_loss)/len(mean_box_loss)
    mean_confidence_loss_out = sum(mean_confidence_loss)/len(mean_confidence_loss)
    mean_noobj_loss_out = sum(mean_noobj_loss)/len(mean_noobj_loss)
    mean_class_loss_out = sum(mean_class_loss)/len(mean_class_loss)  

    print("Total Loss".ljust(12) + "|" + 
          "Box Loss".ljust(12) + "|" + 
          "Conf Loss".ljust(12) + "|" + 
          "No Obj Loss".ljust(12) + "|" + 
          "Class Loss".ljust(12))
    print("------------".ljust(12) + " " + 
          "------------".ljust(12) + " " + 
          "------------".ljust(12) + " " + 
          "------------".ljust(12) + " " + 
          "------------".ljust(12))
    print(f'{train_mean_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_box_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_confidence_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_noobj_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_class_loss_out:.3f}'.ljust(12))
    
    if calculate_mAP == True:
        meanAP = metric.compute()
        metric.reset()  
        print(f'Train mAP = {meanAP["map_50"]:.4f}') 
    else:
        meanAP = {
            'map_50': torch.tensor(0., dtype=torch.float32),
            'map_per_class': torch.tensor([0., 0.], dtype=torch.float32),
            'mar_100_per_class': torch.tensor([0., 0.], dtype=torch.float32), 
        }
        # meanAP = {
        #     'map_50': None,
        #     'map_per_class': None,
        #     'mar_100_per_class': None, 
        # }
        
        
    return (
        {'Total': train_mean_loss_out, 
         'Box': mean_box_loss_out, 
         'Conf': mean_confidence_loss_out, 
         'No Obj': mean_noobj_loss_out, 
         'Class': mean_class_loss_out
        },
        {'mAP': meanAP['map_50'],
         'AP': [meanAP['map_per_class'][0].item(), meanAP['map_per_class'][1].item()],
         'AR': [meanAP['mar_100_per_class'][0].item(), meanAP['mar_100_per_class'][1].item()]
        }
    )
