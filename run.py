import inspect
import sys
import argparse
from pytorch_lightning import Trainer, loggers as pl_loggers
from perception3d.core.config_parser import init_pyinstance
from perception3d.core.config_parser import ConfigParser
from pytorch_lightning.core import LightningModule
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perception 3D main program')
    parser.add_argument('--config_path', required=False, default='configs/run/obj_classification.yaml')
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
    class PerceptionModule(LightningModule):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.phase = kwargs['__phase']
            self.cfg = kwargs
            assert 'model' in kwargs[self.phase], 'must have a model to train/val/test'
            self.model = init_pyinstance(kwargs[self.phase], 'model')
        
        if 'train' in config_args_all:
            def train_dataloader(self):
                return init_pyinstance(self.cfg['train'], 'dataloader')
            
            if 'loss_fn' in config_args_all['train']:
                init_pyinstance(config_args_all['train'], 'loss_fn')
                def training_step(self, batch, batch_idx):
                    preds = self.model(batch)
                    
                    loss_fn = self.cfg['train']['loss_fn']
                    loss = loss_fn(preds, batch)
                    total_loss = sum(loss.values())
                    for k in loss:
                        self.log(k, loss[k].item(), prog_bar=True, on_step=True)
                    
                    for k, pred in preds.items():
                        if isinstance(pred, torch.Tensor):
                            preds[k] = pred.detach()
                    return dict(loss=total_loss, preds=preds, targets=batch)
            
            if 'metric_fn' in config_args_all['train']:
                init_pyinstance(config_args_all['train'], 'metric_fn')
                def training_epoch_end(self, outputs):
                    metric = self.cfg['train']['metric_fn'](preds=outputs['preds'], targets=outputs['targets'])
                    for k in metric:
                        self.log(k, metric[k])
                        
        if 'val' in config_args_all:
            def val_dataloader(self):
                return init_pyinstance(self.cfg['val'], 'dataloader')
            
            def validation_step(self, batch, batch_idx):
                preds = self.model(batch)
                
                if 'loss_fn' in config_args_all['val']:
                    init_pyinstance(config_args_all['val'], 'loss_fn')
                    loss_fn = self.cfg['val']['loss_fn']
                    loss = loss_fn(preds, batch)
                    total_loss = sum(loss.values())
                    for k in loss:
                        self.log(k, loss[k].item(), prog_bar=True, on_step=True)
                    self.log('loss:all', total_loss.item(), prog_bar=True, on_step=True)
                
                for k, pred in preds.items():
                    if isinstance(pred, torch.Tensor):
                        preds[k] = pred.detach()
                return dict(preds=preds, targets=batch)
            
            if 'metric_fn' in config_args_all['val']:
                init_pyinstance(config_args_all['val'], 'metric_fn')
                def validation_epoch_end(self, outputs):
                    metric = self.cfg['val']['metric_fn'](preds=outputs['preds'], targets=outputs['targets'])
                    for k in metric:
                        self.log(k, metric[k])
        
        if 'test' in config_args_all:
            def test_dataloader(self):
                return init_pyinstance(self.cfg['test'], 'dataloader')
            
            def test_step(self, batch, batch_idx):
                preds = self.model(batch)
                
                if 'loss_fn' in config_args_all['test']:
                    init_pyinstance(config_args_all['test'], 'loss_fn')
                    loss_fn = self.cfg['test']['loss_fn']
                    loss = loss_fn(preds, batch)
                    total_loss = sum(loss.values())
                    for k in loss:
                        self.log(k, loss[k].item(), prog_bar=True, on_step=True)
                    self.log('loss:all', total_loss.item(), prog_bar=True, on_step=True)
                
                for k, pred in preds.items():
                    if isinstance(pred, torch.Tensor):
                        preds[k] = pred.detach()
                return dict(preds=preds, targets=batch)
            
            if 'metric_fn' in config_args_all['test']:
                init_pyinstance(config_args_all['test'], 'metric_fn')
                def test_epoch_end(self, outputs):
                    metric = self.cfg['test']['metric_fn'](preds=outputs['preds'], targets=outputs['targets'])
                    for k in metric:
                        self.log(k, metric[k])
        
        def get_progress_bar_dict(self):
            tqdm_dict = super().get_progress_bar_dict()
            tqdm_dict.pop("v_num", None)
            tqdm_dict['loss:all'] = tqdm_dict['loss']
            del tqdm_dict['loss']
            return tqdm_dict

        def configure_optimizers(self):
            opt_cfg = self.cfg['optimizer']
            if not isinstance(opt_cfg, list):
                self.cfg['optimizer'] = [opt_cfg]
            for cfg in self.cfg['optimizer']:
                if 'parameter_scope' in cfg:
                    cfg['params'] = eval('self.' + cfg['parameter_scope'] + '.parameters()')
                    del cfg['parameter_scope']
            opts = init_pyinstance(self.cfg, 'optimizer')
            if 'lr_scheduler' in self.cfg and self.cfg['lr_scheduler'] is not None:
                sched_cfg = self.cfg['lr_scheduler']
                if not isinstance(sched_cfg, list):
                    self.cfg['lr_scheduler'] = [sched_cfg]
                for i, cfg in enumerate(self.cfg['lr_scheduler']):
                    cfg['optimizer'] = opts[i]
                lr_schedulers = init_pyinstance(self.cfg, 'lr_scheduler')
                return opts, lr_schedulers
            return opts
    
    pl_module = PerceptionModule(**config_args_all)
    trainer = Trainer(logger=tb_logger, **trainer_configs)
    
    # run
    if args.phase == 'train':
        trainer.fit(pl_module)
    elif args.phase == 'val':
        trainer.validate(pl_module, pl_module.val_dataloader())
    elif args.phase == 'test':
        trainer.test(pl_module, pl_module.test_dataloader())