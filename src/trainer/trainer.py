from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_decoder import TextDecoder
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker, zero_grad=True, update=True):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train and zero_grad:
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            
            if update:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if not self.is_train:
            decoded_texts = self.text_decoder(**batch)
            batch.update(decoded_texts)

        metrics_split = "train" if self.is_train else "inference"
        for met in self.metrics[metrics_split]:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, spectrogram_orig, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram/augmented", image)
        
        spectrogram_orig_for_plot = spectrogram_orig[0].detach().cpu()
        image_orig = plot_spectrogram(spectrogram_orig_for_plot)
        self.writer.add_image("spectrogram/original", image_orig)

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        logit_metric = TextDecoder(self.text_encoder)

        pred_texts = logit_metric.assemble_predictions(log_probs, log_probs_length)
        targ_texts = logit_metric.transform_targets(text)
        tuples = list(zip(pred_texts, targ_texts, audio_path))

        rows = {
            Path(audio_path).name: {
                "target": target,
                "predictions": pred,
                "wer": calc_wer(target, pred) * 100,
                "cer": calc_cer(target, pred) * 100,
            }
            for pred, target, audio_path in tuples[:examples_to_log]
        }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
