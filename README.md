# crnn_seq2seq_ocr_pytorch

This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN and Sequence to sequence model with attention for image-based sequence recognition tasks, such as scene text recognition and OCR in pytorch using pytorch-lightning.

# Dependencies
All dependencies should be installed are as follows:
* numpy
* opencv-python
* Pillow
* torch
* torchvision
* pytorch-lightning
* jsonargparse[signatures]

Required packages can be installed with
```bash
pip3 install -r requirements.txt
```

# Train a new model


* Create **train_list.txt** and **test_list.txt** as follow format
```
data/dataset/20210420_093714_rst-l0.jpg श्रीशेखावतशक्तसिंहनृपतेमुख्यात्मजाऽप्युत्सुका,
data/dataset/20210420_093714_rst-l1.jpg स्नात्वा-वर्म-विभूष्य-भूषरणपटैर्यानेन-तत्रावजत्-॥२२॥-
data/dataset/20210420_093714_rst-l2.jpg तासां-सत्परिचारिका-निज-निज-स्वामिन्युपस्थायिका-'
data/dataset/20210420_093714_rst-l3.jpg स्तत्रेयुः-किल-या-नृपस्य-च-सदा-छन्दानुगाः-पार्श्वगाः-।-
```

* Start training
```
python3 trainer.py --config default_config.yaml
```

# Help [CLI]

```
➜ python trainer.py -h
usage: trainer.py [-h] [--config CONFIG] [--print_config [={comments,skip_null}+]] [--seed_everything SEED_EVERYTHING] [--trainer.logger LOGGER]
                  [--trainer.checkpoint_callback {true,false}] [--trainer.callbacks CALLBACKS] [--trainer.default_root_dir DEFAULT_ROOT_DIR]
                  [--trainer.gradient_clip_val GRADIENT_CLIP_VAL] [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.process_position PROCESS_POSITION]
                  [--trainer.num_nodes NUM_NODES] [--trainer.num_processes NUM_PROCESSES] [--trainer.gpus GPUS] [--trainer.auto_select_gpus {true,false}]
                  [--trainer.tpu_cores TPU_CORES] [--trainer.ipus IPUS] [--trainer.log_gpu_memory LOG_GPU_MEMORY] [--trainer.progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE]
                  [--trainer.overfit_batches OVERFIT_BATCHES] [--trainer.track_grad_norm TRACK_GRAD_NORM] [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                  [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES] [--trainer.max_epochs MAX_EPOCHS]
                  [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS] [--trainer.max_time MAX_TIME]
                  [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES] [--trainer.limit_val_batches LIMIT_VAL_BATCHES] [--trainer.limit_test_batches LIMIT_TEST_BATCHES]
                  [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                  [--trainer.flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS] [--trainer.accelerator ACCELERATOR]
                  [--trainer.sync_batchnorm {true,false}] [--trainer.precision PRECISION] [--trainer.weights_summary WEIGHTS_SUMMARY] [--trainer.weights_save_path WEIGHTS_SAVE_PATH]
                  [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.truncated_bptt_steps TRUNCATED_BPTT_STEPS]
                  [--trainer.resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--trainer.profiler PROFILER] [--trainer.benchmark {true,false}] [--trainer.deterministic {true,false}]
                  [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS] [--trainer.reload_dataloaders_every_epoch {true,false}]
                  [--trainer.auto_lr_find AUTO_LR_FIND] [--trainer.replace_sampler_ddp {true,false}] [--trainer.terminate_on_nan {true,false}]
                  [--trainer.auto_scale_batch_size AUTO_SCALE_BATCH_SIZE] [--trainer.prepare_data_per_node {true,false}] [--trainer.plugins PLUGINS]
                  [--trainer.amp_backend AMP_BACKEND] [--trainer.amp_level AMP_LEVEL] [--trainer.distributed_backend DISTRIBUTED_BACKEND] [--trainer.move_metrics_to_cpu {true,false}]
                  [--trainer.multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE] [--trainer.stochastic_weight_avg {true,false}] [--model.hidden_size HIDDEN_SIZE]
                  [--model.max_enc_seq_len MAX_ENC_SEQ_LEN] [--model.teaching_forcing_prob TEACHING_FORCING_PROB] [--model.learning_rate LEARNING_RATE] [--model.dropout_p DROPOUT_P]
                  [--model.encoder_pth ENCODER_PTH] [--model.decoder_pth DECODER_PTH] [--model.save_model_dir SAVE_MODEL_DIR] [--data.train_list TRAIN_LIST]
                  [--data.val_list VAL_LIST] [--data.img_height IMG_HEIGHT] [--data.img_width IMG_WIDTH] [--data.num_workers NUM_WORKERS] [--data.batch_size BATCH_SIZE]
                  [--data.random_sampler {true,false}] [--early_stopping.monitor MONITOR] [--early_stopping.min_delta MIN_DELTA] [--early_stopping.patience PATIENCE]
                  [--early_stopping.verbose {true,false}] [--early_stopping.mode MODE] [--early_stopping.strict {true,false}] [--early_stopping.check_finite {true,false}]
                  [--early_stopping.stopping_threshold STOPPING_THRESHOLD] [--early_stopping.divergence_threshold DIVERGENCE_THRESHOLD]
                  [--early_stopping.check_on_train_epoch_end {true,false}] [--model_checkpoint.dirpath DIRPATH] [--model_checkpoint.filename FILENAME]
                  [--model_checkpoint.monitor MONITOR] [--model_checkpoint.verbose {true,false}] [--model_checkpoint.save_last {true,false,null}]
                  [--model_checkpoint.save_top_k SAVE_TOP_K] [--model_checkpoint.save_weights_only {true,false}] [--model_checkpoint.mode MODE]
                  [--model_checkpoint.auto_insert_metric_name {true,false}] [--model_checkpoint.every_n_train_steps EVERY_N_TRAIN_STEPS]
                  [--model_checkpoint.train_time_interval.help CLASS] [--model_checkpoint.train_time_interval TRAIN_TIME_INTERVAL]
                  [--model_checkpoint.every_n_val_epochs EVERY_N_VAL_EPOCHS] [--model_checkpoint.period PERIOD] [--encoder_optimizer.lr LR] [--encoder_optimizer.betas BETAS]
                  [--encoder_optimizer.eps EPS] [--encoder_optimizer.weight_decay WEIGHT_DECAY] [--encoder_optimizer.amsgrad AMSGRAD] [--decoder_optimizer.lr LR]
                  [--decoder_optimizer.betas BETAS] [--decoder_optimizer.eps EPS] [--decoder_optimizer.weight_decay WEIGHT_DECAY] [--decoder_optimizer.amsgrad AMSGRAD]

pytorch-lightning trainer command line tool

optional arguments:
  -h, --help            Show this help message and exit.
  --config CONFIG       Path to a configuration file in json or yaml format.
  --print_config [={comments,skip_null}+]
                        Print configuration and exit.
  --seed_everything SEED_EVERYTHING
                        Set to an int to run seed_everything with this value before classes instantiation (type: Union[int, null], default: null)

Customize every aspect of training via flags:
  --trainer.logger LOGGER
                        Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default ``TensorBoardLogger``. ``False`` will disable logging.
                        (type: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool], default: True)
  --trainer.checkpoint_callback {true,false}
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`. (type: bool, default: True)
  --trainer.callbacks CALLBACKS
                        Add a callback or list of callbacks. (type: Union[List[Callback], Callback, null], default: null)
  --trainer.default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be remote file paths such as `s3://mybucket/path` or
                        'hdfs://path/' (type: Union[str, null], default: null)
  --trainer.gradient_clip_val GRADIENT_CLIP_VAL
                        0 means don't clip. (type: float, default: 0.0)
  --trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM
                        'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm' (type: str, default: norm)
  --trainer.process_position PROCESS_POSITION
                        orders the progress bar when running multiple models on same machine. (type: int, default: 0)
  --trainer.num_nodes NUM_NODES
                        number of GPU nodes for distributed training. (type: int, default: 1)
  --trainer.num_processes NUM_PROCESSES
                        number of processes for distributed training with distributed_backend="ddp_cpu" (type: int, default: 1)
  --trainer.gpus GPUS   number of gpus to train on (int) or which GPUs to train on (list or str) applied per node (type: Union[int, str, List[int], null], default: null)
  --trainer.auto_select_gpus {true,false}
                        If enabled and `gpus` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in "exclusive mode", such
                        that only one process at a time can access them. (type: bool, default: False)
  --trainer.tpu_cores TPU_CORES
                        How many TPU cores to train on (1 or 8) / Single TPU to train on [1] (type: Union[int, str, List[int], null], default: null)
  --trainer.ipus IPUS   How many IPUs to train on. (type: Union[int, null], default: null)
  --trainer.log_gpu_memory LOG_GPU_MEMORY
                        None, 'min_max', 'all'. Might slow performance (type: Union[str, null], default: null)
  --trainer.progress_bar_refresh_rate PROGRESS_BAR_REFRESH_RATE
                        How often to refresh progress bar (in steps). Value ``0`` disables progress bar. Ignored when a custom progress bar is passed to
                        :paramref:`~Trainer.callbacks`. Default: None, means a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.). (type:
                        Union[int, null], default: null)
  --trainer.overfit_batches OVERFIT_BATCHES
                        Overfit a fraction of training data (float) or a set number of batches (int). (type: Union[int, float], default: 0.0)
  --trainer.track_grad_norm TRACK_GRAD_NORM
                        -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm. (type: Union[int, float, str], default: -1)
  --trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Check val every n train epochs. (type: int, default: 5)
  --trainer.fast_dev_run FAST_DEV_RUN
                        runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a sort of unit test). (type: Union[int, bool],
                        default: False)
  --trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates grads every k batches or as set up in the dict. (type: Union[int, Dict[int, int], List[list]], default: 1)
  --trainer.max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and max_steps are not specified, defaults to
                        ``max_epochs`` = 1000. (type: Union[int, null], default: 3)
  --trainer.min_epochs MIN_EPOCHS
                        Force training for at least these many epochs. Disabled by default (None). If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.
                        (type: Union[int, null], default: null)
  --trainer.max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (None). (type: Union[int, null], default: null)
  --trainer.min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (None). (type: Union[int, null], default: null)
  --trainer.max_time MAX_TIME
                        Stop training after this amount of time has passed. Disabled by default (None). The time duration can be specified in the format DD:HH:MM:SS (days, hours,
                        minutes seconds), as a :class:`datetime.timedelta`, or a dictionary with keys that will be passed to :class:`datetime.timedelta`. (type: Union[str, timedelta,
                        Dict[str, int], null], default: null)
  --trainer.limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (float = fraction, int = num_batches) (type: Union[int, float], default: 1.0)
  --trainer.limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (float = fraction, int = num_batches) (type: Union[int, float], default: 1.0)
  --trainer.limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (float = fraction, int = num_batches) (type: Union[int, float], default: 1.0)
  --trainer.limit_predict_batches LIMIT_PREDICT_BATCHES
                        How much of prediction dataset to check (float = fraction, int = num_batches) (type: Union[int, float], default: 1.0)
  --trainer.val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Use float to check within a training epoch, use int to check every n steps (batches). (type: Union[int, float],
                        default: 1.0)
  --trainer.flush_logs_every_n_steps FLUSH_LOGS_EVERY_N_STEPS
                        How often to flush logs to disk (defaults to every 100 steps). (type: int, default: 100)
  --trainer.log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps (defaults to every 50 steps). (type: int, default: 5)
  --trainer.accelerator ACCELERATOR
                        Previously known as distributed_backend (dp, ddp, ddp2, etc...). Can also take in an accelerator object for custom hardware. (type: Union[str, Accelerator,
                        null], default: null)
  --trainer.sync_batchnorm {true,false}
                        Synchronize batch norm layers between process groups/whole world. (type: bool, default: False)
  --trainer.precision PRECISION
                        Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs. (type: int, default: 32)
  --trainer.weights_summary WEIGHTS_SUMMARY
                        Prints a summary of the weights when training begins. (type: Union[str, null], default: top)
  --trainer.weights_save_path WEIGHTS_SAVE_PATH
                        Where to save weights if specified. Will override default_root_dir for checkpoints only. Use this if for whatever reason you need the checkpoints stored in a
                        different place than the logs written in `default_root_dir`. Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/' Defaults to
                        `default_root_dir`. (type: Union[str, null], default: null)
  --trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all batches in all validation dataloaders. (type: int,
                        default: 2)
  --trainer.truncated_bptt_steps TRUNCATED_BPTT_STEPS
                        Deprecated in v1.3 to be removed in 1.5. Please use :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead. (type:
                        Union[int, null], default: null)
  --trainer.resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Path/URL of the checkpoint from which training is resumed. If there is no checkpoint file at the path, start from scratch. If resuming from mid-epoch
                        checkpoint, training will start from the beginning of the next epoch. (type: Union[str, Path, null], default: null)
  --trainer.profiler PROFILER
                        To profile individual steps during training and assist in identifying bottlenecks. (type: Union[BaseProfiler, str, null], default: pytorch)
  --trainer.benchmark {true,false}
                        If true enables cudnn.benchmark. (type: bool, default: True)
  --trainer.deterministic {true,false}
                        If true enables cudnn.deterministic. (type: bool, default: False)
  --trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS
                        Set to a non-negative integer to reload dataloaders every n epochs. Default: 0 (type: int, default: 0)
  --trainer.reload_dataloaders_every_epoch {true,false}
                        Set to True to reload dataloaders every epoch. .. deprecated:: v1.4 ``reload_dataloaders_every_epoch`` has been deprecated in v1.4 and will be removed in
                        v1.6. Please use ``reload_dataloaders_every_n_epochs``. (type: bool, default: False)
  --trainer.auto_lr_find AUTO_LR_FIND
                        If set to True, will make trainer.tune() run a learning rate finder, trying to optimize initial learning for faster convergence. trainer.tune() method will
                        set the suggested learning rate in self.lr or self.learning_rate in the LightningModule. To use a different key set a string instead of True with the key
                        name. (type: Union[bool, str], default: False)
  --trainer.replace_sampler_ddp {true,false}
                        Explicitly enables or disables sampler replacement. If not specified this will toggled automatically when DDP is used. By default it will add ``shuffle=True``
                        for train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it, you can set ``replace_sampler_ddp=False`` and add your own
                        distributed sampler. (type: bool, default: True)
  --trainer.terminate_on_nan {true,false}
                        If set to True, will terminate training (by raising a `ValueError`) at the end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
                        (type: bool, default: False)
  --trainer.auto_scale_batch_size AUTO_SCALE_BATCH_SIZE
                        If set to True, will `initially` run a batch size finder trying to find the largest batch size that fits into memory. The result will be stored in
                        self.batch_size in the LightningModule. Additionally, can be set to either `power` that estimates the batch size through a power search or `binsearch` that
                        estimates the batch size through a binary search. (type: Union[str, bool], default: False)
  --trainer.prepare_data_per_node {true,false}
                        If True, each LOCAL_RANK=0 will call prepare data. Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data (type: bool, default: True)
  --trainer.plugins PLUGINS
                        Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins. (type: Union[List[Union[Plugin, ClusterEnvironment, str]],
                        Plugin, ClusterEnvironment, str, null], default: null)
  --trainer.amp_backend AMP_BACKEND
                        The mixed precision backend to use ("native" or "apex") (type: str, default: native)
  --trainer.amp_level AMP_LEVEL
                        The optimization level to use (O1, O2, etc...). (type: str, default: O2)
  --trainer.distributed_backend DISTRIBUTED_BACKEND
                        deprecated. Please use 'accelerator' (type: Union[str, null], default: null)
  --trainer.move_metrics_to_cpu {true,false}
                        Whether to force internal logged metrics to be moved to cpu. This can save some gpu memory, but can make training slower. Use with attention. (type: bool,
                        default: False)
  --trainer.multiple_trainloader_mode MULTIPLE_TRAINLOADER_MODE
                        How to loop over the datasets when there are multiple train loaders. In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is
                        traversed, and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets reload when reaching the minimum length of
                        datasets. (type: str, default: max_size_cycle)
  --trainer.stochastic_weight_avg {true,false}
                        Whether to use `Stochastic Weight Averaging (SWA) <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_` (type: bool, default:
                        False)

OCR model:
  --model.hidden_size HIDDEN_SIZE
                        size of the lstm hidden state (type: int, default: 256)
  --model.max_enc_seq_len MAX_ENC_SEQ_LEN
                        the width of the feature map out from cnn (type: int, default: 129)
  --model.teaching_forcing_prob TEACHING_FORCING_PROB
                        percentage of samples to apply teach forcing (type: float, default: 0.5)
  --model.learning_rate LEARNING_RATE
                        learning_rate (type: float, default: 0.0001)
  --model.dropout_p DROPOUT_P
                        Dropout probability in Decoder Dropout layer (type: float, default: 0.1)
  --model.encoder_pth ENCODER_PTH
                        path to encoder (to continue training) (type: Union[str, null], default: null)
  --model.decoder_pth DECODER_PTH
                        path to decoder (to continue training) (type: Union[str, null], default: null)
  --model.save_model_dir SAVE_MODEL_DIR
                        Where to store samples and models (type: Union[str, null], default: null)

<class 'aocr.OCRDataModule'>:
  --data.train_list TRAIN_LIST
                        path to train dataset list file (type: Union[str, null], default: data/dataset/train_list.txt)
  --data.val_list VAL_LIST
                        path to validation dataset list file (type: Union[str, null], default: data/dataset/train_list.txt)
  --data.img_height IMG_HEIGHT
                        the height of the input image to network (type: int, default: 32)
  --data.img_width IMG_WIDTH
                        the width of the input image to network (type: int, default: 512)
  --data.num_workers NUM_WORKERS
                        number of data loading num_workers (type: int, default: 2)
  --data.batch_size BATCH_SIZE
                        input batch size (type: int, default: 4)
  --data.random_sampler {true,false}
                        whether to sample the dataset with random sampler (type: bool, default: True)

Linked arguments:
  model.batch_size <-- data.batch_size [applied on parse]
                        input batch size (type: int)
  model.img_height <-- data.img_height [applied on parse]
                        the height of the input image to network (type: int)
  model.img_width <-- data.img_width [applied on parse]
                        the width of the input image to network (type: int)
  model.encoder_optimizer_args <-- add_class_path(encoder_optimizer) [applied on parse]
                        (type: Union[dict, null])
  model.decoder_optimizer_args <-- add_class_path(decoder_optimizer) [applied on parse]
                        (type: Union[dict, null])

Monitor a metric and stop training when it stops improving:
  --early_stopping.monitor MONITOR
                        quantity to be monitored. (type: Union[str, null], default: train_loss)
  --early_stopping.min_delta MIN_DELTA
                        minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than `min_delta`, will count as no improvement. (type:
                        float, default: 0.0)
  --early_stopping.patience PATIENCE
                        number of checks with no improvement after which training will be stopped. Under the default configuration, one check happens after every training epoch.
                        However, the frequency of validation can be modified by setting various parameters on the ``Trainer``, for example ``check_val_every_n_epoch`` and
                        ``val_check_interval``. .. note:: It must be noted that the patience parameter counts the number of validation checks with no improvement, and not the number
                        of training epochs. Therefore, with parameters ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training epochs before
                        being stopped. (type: int, default: 5)
  --early_stopping.verbose {true,false}
                        verbosity mode. (type: bool, default: False)
  --early_stopping.mode MODE
                        one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity monitored has stopped decreasing and in ``'max'`` mode it will stop when
                        the quantity monitored has stopped increasing. (type: str, default: min)
  --early_stopping.strict {true,false}
                        whether to crash the training if `monitor` is not found in the validation metrics. (type: bool, default: True)
  --early_stopping.check_finite {true,false}
                        When set ``True``, stops training when the monitor becomes NaN or infinite. (type: bool, default: True)
  --early_stopping.stopping_threshold STOPPING_THRESHOLD
                        Stop training immediately once the monitored quantity reaches this threshold. (type: Union[float, null], default: null)
  --early_stopping.divergence_threshold DIVERGENCE_THRESHOLD
                        Stop training as soon as the monitored quantity becomes worse than this threshold. (type: Union[float, null], default: null)
  --early_stopping.check_on_train_epoch_end {true,false}
                        whether to run early stopping at the end of the training epoch. If this is ``False``, then the check runs at the end of the validation epoch. (type: bool,
                        default: True)

Save the model periodically by monitoring a quantity. Every metric logged with:
  --model_checkpoint.dirpath DIRPATH
                        directory to save the model file. Example:: # custom path # saves a file like: my/path/epoch=0-step=10.ckpt >>> checkpoint_callback =
                        ModelCheckpoint(dirpath='my/path/') By default, dirpath is ``None`` and will be set at runtime to the location specified by
                        :class:`~pytorch_lightning.trainer.trainer.Trainer`'s :paramref:`~pytorch_lightning.trainer.trainer.Trainer.default_root_dir` or
                        :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_save_path` arguments, and if the Trainer uses a logger, the path will also contain logger name
                        and version. (type: Union[str, Path, null], default: null)
  --model_checkpoint.filename FILENAME
                        checkpoint filename. Can contain named formatting options to be auto-filled. Example:: # save any arbitrary metrics like `val_loss`, etc. in name # saves a
                        file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt >>> checkpoint_callback = ModelCheckpoint( ... dirpath='my/path', ...
                        filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}' ... ) By default, filename is ``None`` and will be set to ``'{epoch}-{step}'``. (type: Union[str, null],
                        default: null)
  --model_checkpoint.monitor MONITOR
                        quantity to monitor. By default it is ``None`` which saves a checkpoint only for the last epoch. (type: Union[str, null], default: train_loss)
  --model_checkpoint.verbose {true,false}
                        verbosity mode. Default: ``False``. (type: bool, default: False)
  --model_checkpoint.save_last {true,false,null}
                        When ``True``, always saves the model at the end of the epoch to a file `last.ckpt`. Default: ``None``. (type: Union[bool, null], default: null)
  --model_checkpoint.save_top_k SAVE_TOP_K
                        if ``save_top_k == k``, the best k models according to the quantity monitored will be saved. if ``save_top_k == 0``, no models are saved. if ``save_top_k ==
                        -1``, all models are saved. Please note that the monitors are checked every ``period`` epochs. if ``save_top_k >= 2`` and the callback is called multiple
                        times inside an epoch, the name of the saved file will be appended with a version count starting with ``v1``. (type: int, default: 1)
  --model_checkpoint.save_weights_only {true,false}
                        if ``True``, then only the model's weights will be saved (``model.save_weights(filepath)``), else the full model is saved (``model.save(filepath)``). (type:
                        bool, default: False)
  --model_checkpoint.mode MODE
                        one of {min, max}. If ``save_top_k != 0``, the decision to overwrite the current save file is made based on either the maximization or the minimization of the
                        monitored quantity. For ``'val_acc'``, this should be ``'max'``, for ``'val_loss'`` this should be ``'min'``, etc. (type: str, default: min)
  --model_checkpoint.auto_insert_metric_name {true,false}
                        When ``True``, the checkpoints filenames will contain the metric name. For example, ``filename='checkpoint_{epoch:02d}-{acc:02d}`` with epoch 1 and acc 80
                        will resolve to ``checkpoint_epoch=01-acc=80.ckp``. Is useful to set it to ``False`` when metric names contain ``/`` as this will result in extra folders.
                        (type: bool, default: True)
  --model_checkpoint.every_n_train_steps EVERY_N_TRAIN_STEPS
                        Number of training steps between checkpoints. If ``every_n_train_steps == None or every_n_train_steps == 0``, we skip saving during training. To disable, set
                        ``every_n_train_steps = 0``. This value must be ``None`` or non-negative. This must be mutually exclusive with ``train_time_interval`` and
                        ``every_n_val_epochs``. (type: Union[int, null], default: null)
  --model_checkpoint.train_time_interval.help CLASS
                        Show the help for the given subclass of timedelta and exit.
  --model_checkpoint.train_time_interval TRAIN_TIME_INTERVAL
                        Checkpoints are monitored at the specified time interval. For all practical purposes, this cannot be smaller than the amount of time it takes to process a
                        single training batch. This is not guaranteed to execute at the exact time specified, but should be close. This must be mutually exclusive with
                        ``every_n_train_steps`` and ``every_n_val_epochs``. (type: Union[timedelta, null], default: null)
  --model_checkpoint.every_n_val_epochs EVERY_N_VAL_EPOCHS
                        Number of validation epochs between checkpoints. If ``every_n_val_epochs == None or every_n_val_epochs == 0``, we skip saving on validation end. To disable,
                        set ``every_n_val_epochs = 0``. This value must be ``None`` or non-negative. This must be mutually exclusive with ``every_n_train_steps`` and
                        ``train_time_interval``. Setting both ``ModelCheckpoint(..., every_n_val_epochs=V)`` and ``Trainer(max_epochs=N, check_val_every_n_epoch=M)`` will only save
                        checkpoints at epochs 0 < E <= N where both values for ``every_n_val_epochs`` and ``check_val_every_n_epoch`` evenly divide E. (type: Union[int, null],
                        default: null)
  --model_checkpoint.period PERIOD
                        Interval (number of epochs) between checkpoints. .. warning:: This argument has been deprecated in v1.3 and will be removed in v1.5. Use
                        ``every_n_val_epochs`` instead. (type: Union[int, null], default: null)

Implements Adam algorithm:
  --encoder_optimizer.lr LR
                        learning rate (default: 1e-3) (type: Any, default: 0.001)
  --encoder_optimizer.betas BETAS
                        coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)) (type: Any, default: (0.9, 0.999))
  --encoder_optimizer.eps EPS
                        term added to the denominator to improve numerical stability (default: 1e-8) (type: Any, default: 1e-08)
  --encoder_optimizer.weight_decay WEIGHT_DECAY
                        weight decay (L2 penalty) (default: 0) (type: Any, default: 0)
  --encoder_optimizer.amsgrad AMSGRAD
                        whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_ (default: False) (type: Any, default: False)

Implements Adam algorithm:
  --decoder_optimizer.lr LR
                        learning rate (default: 1e-3) (type: Any, default: 0.001)
  --decoder_optimizer.betas BETAS
                        coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)) (type: Any, default: (0.9, 0.999))
  --decoder_optimizer.eps EPS
                        term added to the denominator to improve numerical stability (default: 1e-8) (type: Any, default: 1e-08)
  --decoder_optimizer.weight_decay WEIGHT_DECAY
                        weight decay (L2 penalty) (default: 0) (type: Any, default: 0)
  --decoder_optimizer.amsgrad AMSGRAD
                        whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_ (default: False) (type: Any, default: False)

```