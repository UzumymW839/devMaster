import torch
from itertools import repeat
from abc import abstractmethod
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_functions, optimizer, config, epochs=100,
        save_period=1, log_step=2,
        verbosity=2, monitor='off', early_stop=np.inf,
        use_tensorboard=True, tensorboard_log_dir_profiler='saved/log'):

        self.config = config
        self.logger = self.config.get_logger('trainer', verbosity)

        self.model = model
        self.criterion = criterion
        self.metric_functions = metric_functions
        self.optimizer = optimizer

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        self.epochs = epochs
        self.save_period = save_period
        self.monitor = monitor

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
            self.early_stop = early_stop
            if self.early_stop <= 0:
                self.early_stop = np.inf

        self.start_epoch = 1

        self.checkpoint_dir = self.config.save_dir
        self.save_period = save_period
        self.log_step = log_step


        # setup visualization writer instance
        self.writer = SummaryWriter(self.config.log_dir)
        self.tensorboard_log_dir_profiler = tensorboard_log_dir_profiler

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if (epoch % self.save_period == 0) and best:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        best_path = str(self.checkpoint_dir) + '/model_best.pth'
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
