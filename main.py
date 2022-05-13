import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from options import opts
from model import TripletNetwork
from dataloader import SketchyScene, SketchyCOCO, PhotoSketching

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_len, opts.max_len)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # train_dataset = SketchyScene(opts, mode='train', transform=dataset_transforms)
    # val_dataset = SketchyScene(opts, mode='val', transform=dataset_transforms)

    train_dataset = SketchyCOCO(opts, mode='train', transform=dataset_transforms)
    val_dataset = SketchyCOCO(opts, mode='val', transform=dataset_transforms)

    # train_dataset = PhotoSketching(opts, mode='train', transform=dataset_transforms)
    # val_dataset = PhotoSketching(opts, mode='val', transform=dataset_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    model = TripletNetwork().load_from_checkpoint(checkpoint_path="saved_model/deepemd-sketchycoco-epoch=99-top10=0.94.ckpt")
    # model = TripletNetwork().load_from_checkpoint(checkpoint_path="saved_model/deepemd-photosketching-epoch=29-top10=1.00.ckpt")
    # model = TripletNetwork()

    # logger = TensorBoardLogger("tb_logs", name="deepemd-photosketching")

    # checkpoint_callback = ModelCheckpoint(monitor="top5",
    #             mode="max",
    #             dirpath="saved_model",
    #             save_top_k=3,
    #             filename="deepemd-photosketching-{epoch:02d}-{top10:.2f}")

    # trainer = Trainer(gpus=-1, auto_select_gpus=True, # specifies all available GPUs
    #             # auto_scale_batch_size=True,
    #             # auto_lr_find=True,
    #             benchmark=True,
    #             check_val_every_n_epoch=10,
    #             max_epochs=100000,
    #             # precision=64,
    #             min_steps=100, min_epochs=0,
    #             accumulate_grad_batches=8,
    #             # profiler="advanced",
    #             # resume_from_checkpoint="saved_model/deepemd-epoch=119-top10=0.60.ckpt", # "some/path/to/my_checkpoint.ckpt"
    #             logger=logger,
    #             callbacks=[checkpoint_callback])

    # trainer.fit(model, train_loader, val_loader)

    top1 = []
    top10 = []
    for i in range(3):
        trainer = Trainer(logger=False, gpus=-1)
        metrics = trainer.validate(model, val_loader, verbose=False)
        top1.append(metrics[0]['top1'])
        top10.append(metrics[0]['top10'])
    
    print ('Metrics: Top1=%f std=%f, Top10=%f std=%f'%(np.mean(top1), np.std(top1), np.mean(top10), np.std(top10)))
