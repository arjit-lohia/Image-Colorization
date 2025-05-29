# from share import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from colorization_dataset import ColorizationDataset

sd_version = "21"

# Configs
# resume_path = f'./models/control_sd{sd_version}_ini.ckpt'
resume_path = "models/controlnet/v2-1_colorization.ckpt"
batch_size = 1
logger_freq = 30000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(f'models/cldm_v{sd_version}.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = ColorizationDataset("data/colorization/")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False, pin_memory=True)
logger = ImageLogger(batch_frequency=logger_freq)

csv_logger = CSVLogger("logs", name="colorization")

# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints/",               # where to save
#     filename="colorize-{epoch:02d}-{val_loss:.4f}",  # name pattern
#     save_top_k=2,                         # save best 2 checkpoints
#     monitor="val_loss",                  # metric to monitor
#     mode="min",                          # minimize val_loss
#     save_last=True                       # optional: also save last checkpoint
# )

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(csv_logger.log_dir, "checkpoints"),
    filename="last",  # static name to overwrite
    every_n_train_steps=10000,
    save_top_k=1,
    save_last=True
)

trainer = pl.Trainer(
    gpus=1,
    precision=16,
    # callbacks=[logger,checkpoint_callback],
    callbacks=[logger],
    max_epochs=5,
    logger=csv_logger
)
# trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger])
# trainer = pl.Trainer(devices=1, accelerator="cpu", callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
