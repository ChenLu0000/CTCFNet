from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]