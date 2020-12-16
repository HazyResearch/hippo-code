import torch
import pytorch_lightning as pl


def pl_train(cfg, pl_model_class):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
    model = pl_model_class(cfg.model, cfg.dataset, cfg.train)
    if 'pl' in cfg and 'profile' in cfg.pl and cfg.pl.profile:
        # profiler=pl.profiler.AdvancedProfiler(output_filename=cfg.train.profiler),
        profiler_args = { 'profiler': pl.profiler.AdvancedProfiler(), }
    else:
        profiler_args = {}
    if 'pl' in cfg and 'wandb' in cfg.pl and cfg.pl.wandb:
        # kwargs['logger'] = WandbLogger(name=config['pl_wandb'], project='ops-memory-pl')
        logger = WandbLogger(project='ops-memory-pl')
        logger.log_hyperparams(cfg.model)
        logger.log_hyperparams(cfg.dataset)
        logger.log_hyperparams(cfg.train)
        profiler_args['logger'] = logger
    print("profiler args", profiler_args)
    trainer = pl.Trainer(
        # gpus=1 if config['gpu'] else None,
        gpus=1,
        gradient_clip_val=cfg.train.gradient_clip_val,
        max_epochs=1 if cfg.smoke_test else cfg.train.epochs,
        progress_bar_refresh_rate=1,
        limit_train_batches=cfg.train.limit_train_batches,
        track_grad_norm=2,
        **profiler_args,
        logger=False,
    )

    trainer.fit(model)
    # trainer.test(model)
    return trainer, model
