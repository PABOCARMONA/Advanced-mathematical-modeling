<div align="center">

# 🧠 Neural ODEs
### Continuous-depth deep learning through dynamical systems and infinite-dimensional optimal control

<br>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<br>

</div>

---

## 📌 About This Notebook

This notebook gives a self-contained introduction to **Neural ODEs**, combining mathematical rigor with practical implementation. It is organized as follows:

| Step | Section | Description |
|:-:|:---|:---|
| 🔹 | **Intuition** | Neural ODEs are introduced by viewing a residual network as an **explicit Euler scheme** of an ordinary differential equation. |
| 🔹 | **Rigorous derivation** | A rigorous proof of the **backward pass** (adjoint method) using **Lagrangian theory in infinite dimensions** (Hilbert spaces). |
| 🔹 | **Simple implementation** | A first PyTorch implementation on a two-spirals classification problem. |
| 🔹 | **Comparison with ResNet** | A Neural ODE is benchmarked against a discrete residual network on MNIST. |
| 🔹 | **Lorenz system** | A Neural ODE is trained to predict **trajectories of the Lorenz system**, a chaotic dynamical system. |

---

## 🧮 Mathematical Highlights

**💡 Residual networks as Euler schemes.** A residual update

$$y_{n+1} = y_n + f(y_n, \theta)$$

is one explicit Euler step (with $\Delta t = 1$) of the ODE

$$\frac{dy}{dt} = f(y(t), \theta).$$

Letting the step size go to zero produces a model of **continuous depth**, where the forward pass is the numerical solution of an initial value problem.

**📐 Backward pass via infinite-dimensional Lagrangian theory.** The gradient computation is derived from optimal control theory:

- 🔸 State space: $W^{1,2}([t_0,t_1];\mathbb{R}^d)$ (Sobolev space $H^1$)
- 🔸 Constraint space: $L^2([t_0,t_1];\mathbb{R}^d)$
- 🔸 Adjoint state (Lagrange multiplier) identified in $L^2$ via the **Riesz Representation Theorem**

The result is the **adjoint sensitivity method**: gradients come from solving a second ODE backward in time, with $\mathcal{O}(1)$ memory cost.

---

## 🔬 Experiments & Results

### 🌀 Simple Implementation: Two Spirals
A Neural ODE (vector field given by a small MLP, `dopri5` solver) separates two intertwined spirals, reaching about **90–94% test accuracy** on this small synthetic dataset.

### ⚔️ Comparison with ResNet on MNIST
A continuous-depth classifier (**~407k parameters**) is compared with a discrete residual baseline trained under the same setup. The Neural ODE reaches around **98% test accuracy** in 20 epochs, in line with what a ResNet achieves on this task. The main trade-off is **computational cost**: the adaptive solver makes each epoch noticeably slower (~50 s per epoch), in exchange for memory efficiency and a continuous-time interpretation of depth.

### 🦋 Predicting the Lorenz System
A Neural ODE with an encoder–ODE–decoder architecture learns the dynamics of the Lorenz attractor,

$$\dot{x} = \sigma(y-x), \qquad \dot{y} = x(\rho - z) - y, \qquad \dot{z} = xy - \beta z,$$

with $\sigma = 10$, $\rho = 28$, $\beta = 8/3$. From an initial state, the model predicts the full trajectory, and results are evaluated on unseen initial conditions with 3D trajectory visualizations. Neural ODEs are particularly well suited to **irregularly sampled time series**, where classical recurrent architectures struggle.

---

## ⚖️ Strengths & Limitations

| ✅ Strengths | ⚠️ Limitations |
|:---|:---|
| **Parametric efficiency**: weights are shared along the whole trajectory | **Topological constraints**: by Picard–Lindelöf, trajectories cannot cross (remedy: *Augmented Neural ODEs*) |
| **$\mathcal{O}(1)$ memory** thanks to the adjoint method | **Backward instability** under strongly dissipative dynamics |
| **Natural fit for irregular time series** | **Inefficient on purely static data** compared with CNNs or ViTs |

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)

</div>

**Main libraries:** `torch` · `torchdiffeq` · `numpy` · `scipy` · `scikit-learn` · `matplotlib` · `seaborn`

---








