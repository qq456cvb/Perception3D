import inspect
import sys
import argparse
from pytorch_lightning import Trainer, loggers as pl_loggers
from perception3d.core.config_parser import init_pyinstance
from perception3d.core.config_parser import ConfigParser
from pytorch_lightning.core import LightningModule
import torch

class PerceptionModule(LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.phase = kwargs['__phase']
        self.cfg = kwargs
        assert 'model' in kwargs[self.phase], 'must have a model to train/val/test'
        init_pyinstance(kwargs[self.phase]['model'], kwargs[self.phase], 'model')
        self.model = kwargs[self.phase]['model']
    
    def train_dataloader(self):
        init_pyinstance(self.cfg['train']['dataloader'], self.cfg['train'], 'dataloader')
        return self.cfg['train']['dataloader']
    
    def val_dataloader(self):
        init_pyinstance(self.cfg['val']['dataloader'], self.cfg['val'], 'dataloader')
        return self.cfg['val']['dataloader']
    
    def test_dataloader(self):
        init_pyinstance(self.cfg['test']['dataloader'], self.cfg['test'], 'dataloader') 
        return self.cfg['test']['dataloader']
    
    def training_step(self, batch, batch_idx):
        preds = self.model(batch)
        assert 'loss_fn' in self.cfg['train'], 'must implement the loss function in order to train'
        init_pyinstance(self.cfg['train']['loss_fn'], self.cfg['train'], 'loss_fn')
        loss_fn = self.cfg['train']['loss_fn']
        loss = loss_fn(preds, batch)
        for k in loss:
            self.log(k, loss[k].item(), prog_bar=True, on_step=True)
        return dict(loss=loss, preds=preds, targets=batch)
    
    def training_epoch_end(self, outputs):
        if 'metric_fn' in self.cfg['train']:
            init_pyinstance(self.cfg['train']['metric_fn'], self.cfg['train'], 'metric_fn')
            metric_fn = self.cfg['train']['metric_fn']
            metric = metric_fn(preds=outputs['preds'], targets=outputs['targets'])
            for k in metric:
                self.log(k, metric[k], on_epoch=True)
        return super().training_epoch_end(outputs)
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perception 3D main program')
    parser.add_argument('--config_path', required=False, default='configs/models/classification/object/pointnet2.yaml')
    parser.add_argument('--phase', required=False, choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    
    config_args_all = ConfigParser().parse(args.config_path)
    config_args_all['__phase'] = args.phase
    
    # if define a global model
    if 'model' in config_args_all:
        for phase in ['train', 'val', 'test']:
            config_args_all[phase]['model'] = config_args_all['model']
        del config_args_all['model']
    config_args = config_args_all[args.phase]
    
    tb_logger = pl_loggers.TensorBoardLogger(config_args['log_dir'])
    trainer_keys = [p for p in inspect.signature(Trainer.__init__).parameters if p != 'self']
    trainer_configs = dict([(k, config_args[k]) for k in config_args if k in trainer_keys])
    
    # init
    pl_module = PerceptionModule(**config_args_all)
    trainer = Trainer(logger=tb_logger, **trainer_configs)
    
    # run
    if args.phase == 'train':
        trainer.fit(pl_module)
    elif args.phase == 'val':
        trainer.validate(pl_module, pl_module.val_dataloader())
    elif args.phase == 'test':
        trainer.test(pl_module, pl_module.test_dataloader())