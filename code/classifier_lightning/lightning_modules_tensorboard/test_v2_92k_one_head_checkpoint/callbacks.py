from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    dirpath="results/",
    filename='{epoch}-{val_smoke_pre:.2f}-{val_smoke_f1:.2f}-{val_fire_f1:.2f}',
    save_top_k=5, 
    mode='max',
    monitor="val_smoke_pre")

