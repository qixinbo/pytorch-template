import torchmetrics
from pytorch_accelerated.callbacks import TrainerCallback

class accuracy(TrainerCallback):
    def __init__(self):
        self.accuracy = torchmetrics.Accuracy()

    def on_training_run_start(self, trainer, **kwargs):
        self.accuracy.to(trainer.device)

    def on_eval_step_end(self, trainer, batch, batch_output, **kwargs):
        preds = batch_output["model_outputs"].argmax(dim=-1)
        self.accuracy.update(preds, batch[1])

    def on_eval_epoch_end(self, trainer, **kwargs):
        trainer.run_history.update_metric("accuracy", self.accuracy.compute().item())
        self.accuracy.reset()
