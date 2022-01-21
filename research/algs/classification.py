import torch

from .base import Algorithm

class Classification(Algorithm):

    def __init__(self, env, network_class, dataset_class, **kwargs):
        super().__init__(env, network_class, dataset_class, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _compute_loss_and_accuracy(self, batch):
        '''
        Helper function used to reduce repeated computation
        '''
        x, y = batch
        yhat = self.network(x)
        loss = self.criterion(yhat, y)
        with torch.no_grad():
            accuracy = (torch.argmax(yhat, dim=-1) == y).sum() / y.shape[0]
        return loss, accuracy

    def _train_step(self, batch):
        '''
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of training metrics.
        '''
        self.optim['network'].zero_grad() # By default, the optimizer is stored under 'network' if no custom optimizer setup is performed.
        loss, accuracy = self._compute_loss_and_accuracy(batch)
        loss.backward()
        self.optim['network'].step()
        return dict(loss=loss.item(), accuracy=accuracy.item())

    def _validation_step(self, batch):
        '''
        Overriding the Algorithm BaseClass Method _train_step.
        Returns a dictionary of validation metrics.
        '''
        with torch.no_grad():
            loss, accuracy = self._compute_loss_and_accuracy(batch)
        return dict(loss=loss.item(), accuracy=accuracy.item())
