import torch
import pathlib

from trainers.base_trainer import BaseTrainer
from utils.metric_tracker import MetricTracker
from utils.training_utils import save_maxscaled_audio, save_specgram_fig
from utils.running_average import RunningAverage


class AudioTrainer(BaseTrainer):

    def __init__(self, device, dataloader, valid_dataloader=None, lr_scheduler=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.dataloader = dataloader

        # epoch-based training
        self.len_epoch = len(self.dataloader)

        self.global_step = 0

        self.valid_dataloader = valid_dataloader
        self.do_validation = self.valid_dataloader is not None
        self.lr_scheduler = lr_scheduler

        self.valid_metrics = MetricTracker('val_loss', 'avg_val_loss', 'avg_train_loss', *[m.__name__ for m in self.metric_functions], writer=self.writer)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=self.global_step)

        self.audio_example_path = pathlib.Path(self.checkpoint_dir).joinpath('best_audio_examples')
        self.audio_example_path.mkdir(exist_ok=False) # do not overwrite, it could be a good example!?


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        loss_avg = RunningAverage()

        for batch_idx, (data, target) in enumerate(self.dataloader):

            self.optimizer.zero_grad()
            for param in self.model.parameters():
                param.grad = None

            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

            with torch.autocast("cuda"):
                output = self.model(data)
                loss = self.criterion(output, target)

            self.scaler.scale(loss).backward()
            loss_avg.update(loss.detach())


            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            if batch_idx % self.log_step == 0:

                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.detach()))

            if batch_idx == self.len_epoch: # epoch stop # TODO: is this required? does this break anything?
                break

        self.valid_metrics.update('avg_train_loss', loss_avg(), global_step=epoch)

        if self.do_validation:
            val_log = self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log["val_loss"])
        return val_log


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        loss_avg = RunningAverage()


        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_dataloader):

                data, target = data.to(self.device), target.to(self.device)

                with torch.autocast("cuda"):
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.valid_metrics.update('val_loss', loss.detach(), global_step=self.global_step)
                loss_avg.update(loss.detach())
            # compute metrics on the last validation file and save it as audio
            self.compute_all_metrics(output=output[-1, ...], target=target[-1, ...])
            self.export_example(output=output[-1, ...], target=target[-1, ...], data=data[-1,...])


        self.valid_metrics.update('avg_val_loss', loss_avg(), global_step=epoch)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto', global_step=self.global_step)
        return self.valid_metrics.result()


    def compute_all_metrics(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """
        compute metrics for each example and then average.
        Parameters:
            output: Tensor containing the output of the model.
            target: the desired output as tensor.
        """
        for met in self.metric_functions:
            result_list = []
            for out, targ in zip(output, target):
                result_list.append(met(out, targ))
            result = torch.nanmean(torch.tensor(result_list))

            self.valid_metrics.update(met.__name__, result, global_step=self.global_step)


    def _progress(self, batch_idx: int) -> str:
        """
        Foramt a string indicating the progress in the current epoch.
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.dataloader, 'n_samples'):
            current = batch_idx * self.dataloader.batch_size
            total = self.dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    def export_example(self, output: torch.Tensor, target: torch.Tensor, data: torch.Tensor):

        with torch.no_grad():

            inp = data.detach().cpu() # select target reference channel
            tar = target.detach().cpu() # target is single-channel anyway
            out = output.detach().cpu() # target is single-channel anyway

            # write audio and spectrograms to tensorboard log files
            for inp_ch in range(inp.shape[0]):
                # save each channel of the input as a separate spectrogram
                save_specgram_fig(self.writer, self.global_step, inp[inp_ch, :], f'noisy input ch.{inp_ch}')
            save_specgram_fig(self.writer, self.global_step, tar[0, :], 'target')
            save_specgram_fig(self.writer, self.global_step, out[0, :], 'output')

            # save single- and multi-channel audio example
            save_maxscaled_audio(inp, audio_name_tag='noisy input', audio_example_path=self.audio_example_path)
            save_maxscaled_audio(tar, audio_name_tag='target', audio_example_path=self.audio_example_path)
            save_maxscaled_audio(out, audio_name_tag='output', audio_example_path=self.audio_example_path)

