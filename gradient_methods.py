import logging
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

lock = threading.Lock()


class GradientOptimizer:
    """
    Gradient-based optimizer class providing various optimization algorithms.

    ...

    Attributes
    ----------
    model : nn.Module
        The model to optimize.
    lr : float
        Learning rate for the optimizer.
    velocity : torch.Tensor
        Velocity vector for algorithms like momentum and Adam.
    beta : float
        Exponential decay rate for the moving average of past gradients.
    beta1, beta2 : float
        Exponential decay rates for Adam optimizer.
    epsilon : float
        Small value to avoid division by zero in Adam optimizer.
    weight_decay : float
        Weight decay (L2 penalty) coefficient.
    momentum : float
        Momentum coefficient for momentum-based optimizers.
    device : torch.device
        Device on which the tensors are stored.
    params : list
        List of model parameters to optimize.

    Methods
    -------
    optimize(loss_fn, data_loader, epochs, weight_decay=0.0, momentum=0.0, algorithm='sgd')
        Perform optimization using the specified algorithm.
    sgd(grad, velocity, lr, weight_decay, momentum, params)
        Update parameters using the Stochastic Gradient Descent algorithm.
    momentum(grad, velocity, lr, beta, weight_decay, momentum, params)
        Update parameters using the Momentum optimization algorithm.
    adam(grad, velocity, lr, beta1, beta2, epsilon, weight_decay, momentum, params)
        Update parameters using the Adam optimization algorithm.
    step(velocity, beta, beta1, beta2, params)
        Update velocity and parameters based on the optimizer algorithm.
    update_velocity(grad, velocity, beta, beta1, beta2)
        Update the velocity vector for momentum-based optimizers.
    clip_gradients(grad, clip_value)
        Clip gradients to a specified value to prevent explosion.
    backprop(loss, model)
        Perform backward pass and compute gradients.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001, beta: float = 0.9, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0, momentum: float = 0.0, device: str = 'cpu'):
        """
        Initialize the GradientOptimizer with the model and optimization parameters.

        Parameters
        ----------
        model : nn.Module
            The model to optimize.
        lr : float, optional
            Learning rate for the optimizer (default: 0.001).
        beta : float, optional
            Exponential decay rate for the moving average of past gradients (default: 0.9).
        beta1 : float, optional
            Exponential decay rate for the moving average of past gradients in Adam (default: 0.9).
        beta2 : float, optional
            Exponential decay rate for the moving average of squared gradients in Adam (default: 0.999).
        epsilon : float, optional
            Small value to avoid division by zero in Adam optimizer (default: 1e-8).
        weight_decay : float, optional
            Weight decay (L2 penalty) coefficient (default: 0).
        momentum : float, optional
            Momentum coefficient for momentum-based optimizers (default: 0.0).
        device : str, optional
            Device on which the tensors are stored (default: 'cpu').
        """
        self.model = model
        self.lr = lr
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = torch.device(device)
        self.params = [param for param in model.parameters() if param.requires_grad]
        self.velocity = None

    def optimize(self, loss_fn, data_loader: DataLoader, epochs: int, weight_decay: float = 0.0, momentum: float = 0.0,
                 algorithm: str = 'sgd'):
        """
        Perform optimization using the specified algorithm.

        Parameters
        ----------
        loss_fn : callable
            The loss function to minimize.
        data_loader : DataLoader
            Data loader providing batches of input data and corresponding labels.
        epochs : int
            Number of epochs to run the optimization.
        weight_decay : float, optional
            Weight decay (L2 penalty) coefficient (overrides optimizer value).
        momentum : float, optional
            Momentum coefficient (overrides optimizer value) for momentum-based optimizers.
        algorithm : str, optional
            Optimization algorithm to use (sgd, momentum, adam).

        Returns
        -------
        None
        """
        self.velocity = torch.zeros_like(self.params[0], device=self.device) if self.velocity is None else self.velocity

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Compute loss and gradients
                loss = loss_fn(self.model(data), target)
                total_loss += loss.item()

                self.backprop(loss, self.model)

                # Apply gradient-based optimization
                if algorithm == 'sgd':
                    self.sgd(self.model.grad, self.velocity, self.lr, weight_decay, momentum, self.params)
                elif algorithm == 'momentum':
                    self.momentum(self.model.grad, self.velocity, self.lr, self.beta, weight_decay, momentum, self.params)
                elif algorithm == 'adam':
                    self.adam(self.model.grad, self.velocity, self.lr, self.beta1, self.beta2, self.epsilon, weight_decay, momentum,
                              self.params)
                else:
                    raise ValueError(f"Unsupported optimization algorithm: {algorithm}")

                # Zero gradients after updating weights
                self.model.zero_grad()

            # Log epoch loss
            logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader):.4f}")

    def sgd(self, grad: torch.Tensor, velocity: torch.Tensor, lr: float, weight_decay: float, momentum: float, params: list):
        """
        Update parameters using the Stochastic Gradient Descent algorithm.

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the model parameters.
        velocity : torch.Tensor
            Velocity vector (not used in SGD).
        lr : float
            Learning rate.
        weight_decay : float
            Weight decay (L2 penalty) coefficient.
        momentum : float
            Momentum coefficient (not used in SGD).
        params : list
            List of model parameters to update.

        Returns
        -------
        None
        """
        with lock:
            for param in params:
                param.data -= lr * (grad[param] + weight_decay * param)

    def momentum(self, grad: torch.Tensor, velocity: torch.Tensor, lr: float, beta: float, weight_decay: float, momentum: float,
                 params: list):
        """
        Update parameters using the Momentum optimization algorithm.

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the model parameters.
        velocity : torch.Tensor
            Velocity vector for momentum.
        lr : float
            Learning rate.
        beta : float
            Exponential decay rate for the moving average of past gradients.
        weight_decay : float
            Weight decay (L2 penalty) coefficient.
        momentum : float
            Momentum coefficient.
        params : list
            List of model parameters to update.

        Returns
        -------
        None
        """
        with lock:
            self.step(velocity, beta, 0, 0, params)
            for param in params:
                param.data -= lr * (velocity[param] + weight_decay * param)

    def adam(self, grad: torch.Tensor, velocity: torch.Tensor, lr: float, beta1: float, beta2: float, epsilon: float,
             weight_decay: float, momentum: float, params: list):
        """
        Update parameters using the Adam optimization algorithm.

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the model parameters.
        velocity : torch.Tensor
            Velocity vector for Adam.
        lr : float
            Learning rate.
        beta1 : float
            Exponential decay rate for the moving average of past gradients.
        beta2 : float
            Exponential decay rate for the moving average of squared gradients.
        epsilon : float
            Small value to avoid division by zero.
        weight_decay : float
            Weight decay (L2 penalty) coefficient.
        momentum : float
            Momentum coefficient.
        params : list
            List of model parameters to update.

        Returns
        -------
        None
        """
        t = np.sqrt(1 - beta2**self.t) / (1 - beta1**self.t)
        self.t += 1

        with lock:
            self.step(velocity, beta1, beta2, epsilon, params)
            for param in params:
                param.data -= lr * t * (velocity[param] / (torch.sqrt(velocity[param**2]) + epsilon) + weight_decay * param)

    def step(self, velocity: torch.Tensor, beta: float, beta1: float, beta2: float, params: list):
        """
        Update velocity and parameters based on the optimizer algorithm.

        Parameters
        ----------
        velocity : torch.Tensor
            Velocity vector to update.
        beta : float
            Exponential decay rate for the moving average of past gradients.
        beta1 : float
            Exponential decay rate for the moving average of past gradients in Adam.
        beta2 : float
            Exponential decay rate for the moving average of squared gradients in Adam.
        params : list
            List of model parameters.

        Returns
        -------
        None
        """
        with lock:
            for param in params:
                velocity[param] = beta * velocity[param] + (1 - beta) * grad[param]
                if beta1 != 0:
                    velocity[param] /= (1 - beta1**self.t)
                if beta2 != 0:
                    velocity[param**2] = beta2 * velocity[param**2] + (1 - beta2) * (grad[param]**2)

    def update_velocity(self, grad: torch.Tensor, velocity: torch.Tensor, beta: float, beta1: float, beta2: float):
        """
        Update the velocity vector for momentum-based optimizers.

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the model parameters.
        velocity : torch.Tensor
            Velocity vector to update.
        beta : float
            Exponential decay rate for the moving average of past gradients.
        beta1 : float
            Exponential decay rate for the moving average of past gradients in Adam.
        beta2 : float
            Exponential decay rate for the moving average of squared gradients in Adam.

        Returns
        -------
        None
        """
        with lock:
            for param in self.params:
                velocity[param] = beta * velocity[param] + (1 - beta) * grad[param]
                if beta1 != 0:
                    velocity[param] *= beta1
                if beta2 != 0:
                    velocity[param**2] = beta2 * velocity[param**2] + (1 - beta2) * (grad[param]**2)

    def clip_gradients(self, grad: torch.Tensor, clip_value: float):
        """
        Clip gradients to a specified value to prevent explosion.

        Parameters
        ----------
        grad : torch.Tensor
            Gradients of the model parameters.
        clip_value : float
            Maximum gradient value allowed.

        Returns
        -------
        None
        """
        with lock:
            for param in self.params:
                grad[param].data.clamp_(-clip_value, clip_value)

    def backprop(self, loss: torch.Tensor, model: nn.Module):
        """
        Perform backward pass and compute gradients.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor.
        model : nn.Module
            Model for which gradients are computed.

        Returns
        -------
        None
        """
        loss.backward()


class GradientOptimizerV2(GradientOptimizer):
    """
    Extended GradientOptimizer with additional functionality.

    ...

    Attributes
    ----------
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.
    grad_clipping : bool
        Flag indicating if gradient clipping is enabled.
    clip_value : float
        Maximum gradient value allowed for clipping.

    Methods
    -------
    optimize_with_scheduler(loss_fn, data_loader, epochs, scheduler, weight_decay=0.0, momentum=0.0, algorithm='sgd')
        Perform optimization using a learning rate scheduler.
    """

    def __init__(self, model: nn.Module, lr: float = 0.001, beta: float = 0.9, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8, weight_decay: float = 0, momentum: float = 0.0, device: str = 'cpu', lr_scheduler=None,
                 grad_clipping: bool = False, clip_value: float = 1.0):
        """
        Initialize the GradientOptimizerV2 with extended parameters.

        Parameters
        ----------
        model : nn.Module
            The model to optimize.
        lr : float, optional
            Learning rate for the optimizer (default: 0.001).
        beta : float, optional
            Exponential decay rate for the moving average of past gradients (default: 0.9).
        beta1 : float, optional
            Exponential decay rate for the moving average of past gradients in Adam (default: 0.9).
        beta2 : float, optional
            Exponential decay rate for the moving average of squared gradients in Adam (default: 0.999).
        epsilon : float, optional
            Small value to avoid division by zero in Adam optimizer (default: 1e-8).
        weight_decay : float, optional
            Weight decay (L2 penalty) coefficient (default: 0).
        momentum : float, optional
            Momentum coefficient for momentum-based optimizers (default: 0.0).
        device : str, optional
            Device on which the tensors are stored (default: 'cpu').
        lr_scheduler : torch.optim.lr_scheduler, optional
            Learning rate scheduler (default: None).
        grad_clipping : bool, optional
            Flag indicating if gradient clipping is enabled (default: False).
        clip_value : float, optional
            Maximum gradient value allowed for clipping (default: 1.0).
        """
        super().__init__(model, lr, beta, beta1, beta2, epsilon, weight_decay, momentum, device)
        self.lr_scheduler = lr_scheduler
        self.grad_clipping = grad_clipping
        self.clip_value = clip_value

    def optimize_with_scheduler(self, loss_fn, data_loader: DataLoader, epochs: int, scheduler: torch.optim.lr_scheduler,
                               weight_decay: float = 0.0, momentum: float = 0.0, algorithm: str = 'sgd'):
        """
        Perform optimization using a learning rate scheduler.

        Parameters
        ----------
        loss_fn : callable
            The loss function to minimize.
        data_loader : DataLoader
            Data loader providing batches of input data and corresponding labels.
        epochs : int
            Number of epochs to run the optimization.
        scheduler : torch.optim.lr_scheduler
            Learning rate scheduler.
        weight_decay : float, optional
            Weight decay (L2 penalty) coefficient (overrides optimizer value).
        momentum : float, optional
            Momentum coefficient (overrides optimizer value) for momentum-based optimizers.
        algorithm : str, optional
            Optimization algorithm to use (sgd, momentum, adam).

        Returns
        -------
        None
        """
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Compute loss and gradients
                loss = loss_fn(self.model(data), target)
                total_loss += loss.item()

                self.backprop(loss, self.model)

                # Gradient clipping
                if self.grad_clipping:
                    self.clip_gradients(self.model.grad, self.clip_value)

                # Apply gradient-based optimization
                if algorithm == 'sgd':
                    self.sgd(self.model.grad, self.velocity, self.lr, weight_decay, momentum, self.params)
                elif algorithm == 'momentum':
                    self.momentum(self.model.grad, self.velocity, self.lr, self.beta, weight_decay, momentum, self.params)
                elif algorithm == 'adam':
                    self.adam(self.model.grad, self.velocity, self.lr, self.beta1, self.beta2, self.epsilon, weight_decay, momentum,
                              self.params)
                else:
                    raise ValueError(f"Unsupported optimization algorithm: {algorithm}")

                # Update learning rate
                scheduler.step()

                # Zero gradients after updating weights
                self.model.zero_grad()

            # Log epoch loss
            logger.info(f"Epoch {epoch+1} Loss: {total_loss / len(data_loader):.4f}")


def build_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    """
    Build a PyTorch DataLoader for the input data and labels.

    Parameters
    ----------
    data : np.ndarray
        Input data samples.
    labels : np.ndarray
        Corresponding labels for the data samples.
    batch_size : int, optional
        Number of samples per batch (default: 32).
    shuffle : bool, optional
        Flag indicating if the data should be shuffled (default: True).

    Returns
    -------
    DataLoader
        PyTorch DataLoader for the input data and labels.
    """