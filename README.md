# Gradient Descent in PyTorch

![Gradient Descent gif](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Gradient_descent.gif/640px-Gradient_descent.gif)

## Introduction

Gradient Descent is an optimization algorithm used to minimize a differentiable function. It's widely used in machine learning and deep learning to find the minimum of a loss or cost function.

## Objective

Minimize a differentiable function $f(x)$ with respect to a parameter vector $x$.

## Algorithm

### Initialization
1. Start with an initial guess for the parameter vector $x$, denoted as $x_0$.
2. Choose a learning rate $\alpha$, which determines the step size in the parameter space.

### Iteration
For each iteration $t$ update the parameter vector $x$ as follows:
$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$
where:<br>
- $x_t$ is the parameter vector at iteration $t$.
- $\nabla f(x_t)$ is the gradient of the function $f(x)$ at $x_t$, which is a vector contraining the partial derivatives of $f$ with respect to each component of $x$. Mathematically, it is expressed as:
$$\nabla f(x_t) = \Bigg [ \frac{\partial{f(x)}}{\partial{x_1}}, \frac{\partial{f(x)}}{\partial{x_2}},..., \frac{\partial{f(x)}}{\partial{x_n}}  \Bigg ]$$
- $\alpha$ is the learning rate, a hyperparameter that controls the step size in the parameter space. It is typicaly a small positive value.

### Termination
Repeat the iteration step until one of the stopping criteria is met, such as a maximum number of iterations or when the change in the objective function becomes sufficiently small.


## Explanation
In each iteration, Gradient Descent computes the gradient $\nabla f(x_t)$ of the function $f(x)$ at the current parameter vector $x_t$. The gradient points in the direction of the steepest increase in the function.

It then updates the parameter vector $x$ by subtracting the gradient scaled by the learning rate $\alpha$. This update moves $x$ in the direction that reduces the value of the function.

The learning rate $\alpha$ controls the step size. A larger $\alpha$ can lead to faster convergence, but it can also cause overshooting and divergence. A smaller $\alpha$ can make the convergence more stable but slower.

The algorithm repeats this process iteratively, gradually moving towards the minimum of the function.

The process stops when a termination condition is met, such as reaching a maximum number of iterations or when the change in the objective function becomes very small.


## Implementation
In this project, the gradient descent algorithm is demonstrated using a simple loss function using PyTorch's autograd capabilities.

The function $f(x) = x^2 + 4x + 4$ is used as the loss function and the parameter $x$ is randomly initialized. Mathematically, the minimum of this function occurs at $x=2$. We use `backward()` method provided by PyTorch to compute gradients associated with the tensor and then visualize the convergence of loss to the global minima. 
