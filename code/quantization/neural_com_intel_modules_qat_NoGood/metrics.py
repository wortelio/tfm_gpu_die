import torchmetrics
import config

precision_metric = torchmetrics.classification.MultilabelPrecision(num_labels = config.N_CLASSES, 
                                                                   threshold = 0.5, 
                                                                   average = None).to(config.DEVICE)
recall_metric = torchmetrics.classification.MultilabelRecall(num_labels = config.N_CLASSES, 
                                                             threshold = 0.5, 
                                                             average = None).to(config.DEVICE)
accuracy_metric = torchmetrics.classification.MultilabelAccuracy(num_labels = config.N_CLASSES, 
                                                                 threshold = 0.5, 
                                                                 average = None).to(config.DEVICE)
f1_metric = torchmetrics.classification.MultilabelF1Score(num_labels = config.N_CLASSES, 
                                                          threshold = 0.5, 
                                                          average = None).to(config.DEVICE)

