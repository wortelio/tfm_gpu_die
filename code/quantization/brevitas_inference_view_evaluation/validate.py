from tqdm import tqdm
import metrics
import torch

def eval_fn(loader, model, device):
    
    model.eval()
    loop = tqdm(loader, desc='Validating', leave=True)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
      
        yhat = torch.sigmoid(out.detach())
        metrics.precision_metric.update(yhat, y)
        metrics.recall_metric.update(yhat, y)
        metrics.accuracy_metric.update(yhat, y)
        metrics.f1_metric.update(yhat, y)
        metrics.f1_metric_mean.update(yhat, y)
    
    precision = metrics.precision_metric.compute()
    recall = metrics.recall_metric.compute()
    accuracy = metrics.accuracy_metric.compute()
    f1 = metrics.f1_metric.compute()
    f1_mean = metrics.f1_metric_mean.compute()
    
    metrics.precision_metric.reset()
    metrics.recall_metric.reset()
    metrics.accuracy_metric.reset()
    metrics.f1_metric.reset()
    metrics.f1_metric_mean.reset()

    print(f'SMOKE -> Precision: {precision[0]:.4f} - Recall: {recall[0]:.4f} - Accuracy: {accuracy[0]:.4f} - F1: {f1[0]:.4f}')
    print(f'FIRE -> Precision: {precision[1]:.4f} - Recall: {recall[1]:.4f} - Accuracy: {accuracy[1]:.4f} - F1: {f1[1]:.4f}')
    print(f'Mean F1 Score: {f1_mean.item():.4f}')
    
    return (
        {
        'Accuracy': [accuracy[0].item(), accuracy[1].item()],
        'Precision': [precision[0].item(), precision[1].item()],
        'Recall': [recall[0].item(), recall[1].item()],
        'F1': [f1[0].item(), f1[1].item()],
        'F1 mean': f1_mean.item(),
        }
    )