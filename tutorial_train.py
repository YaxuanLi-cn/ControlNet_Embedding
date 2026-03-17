from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd15_normal.pth'
dataset_path = './pairUAV/train'
img_dir = './pairUAV/tours/'
batch_size = 4
logger_freq = 10000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(dataset_path=dataset_path, img_dir=img_dir)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
checkpoint_callback = ModelCheckpoint(every_n_train_steps=logger_freq, save_top_k=0, save_last=True)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], max_epochs=1)


# Train!
trainer.fit(model, dataloader)
