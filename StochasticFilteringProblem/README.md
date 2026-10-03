<div align="center">

# 📡 Filtering Problems
### Optimal state estimation in stochastic systems using continuous Kalman-Bucy and Particle filters

<br>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<br>

</div>

---

## 📌 About This Notebook

This notebook provides a comprehensive introduction to **Stochastic Filtering**, bridging the gap between rigorous stochastic calculus and practical computational algorithms. It is organized as follows:

| Step | Section | Description |
|:-:|:---|:---|
| 🔹 | **The Filtering Problem** | Formalizing hidden states and noisy observations as continuous-time **Stochastic Differential Equations (SDEs)**. |
| 🔹 | **1D Kalman-Bucy Filter** | Exact continuous-time optimal filtering for linear systems using the deterministic **Riccati equation**. |
| 🔹 | **Multi-dimensional Kalman** | Extending the theory to higher dimensions with matrix SDEs and covariance tracking. |
| 🔹 | **Particle Filters (SMC)** | Solving intractable non-linear filtering problems numerically using **Sequential Monte Carlo** and the SIR algorithm. |

---

## 🧮 Mathematical Highlights

**💡 The Continuous Linear Filter.** For a linear system $dX_t = F(t)X_t dt + C(t)dU_t$ with linear observations $dZ_t = G(t)X_t dt + D(t)dV_t$, the optimal estimate $\widehat{X}_t$ evolves according to the stochastic differential equation:

$$d\widehat{X}_t = \left(F - SG^T(DD^T)^{-1}G \right)\widehat{X}_t dt + SG^T(DD^T)^{-1}dZ_t$$

where the error covariance $S(t)$ is exactly solved by the **matrix Riccati equation**.

**📐 Non-Linear Filtering via Empirical Measures.** When the system is highly non-linear or non-Gaussian, the conditional probability density $\pi_t(x)$ is infinite-dimensional. Particle filters approximate it using a weighted sum of Dirac delta functions:

$$\pi_t(x) \approx \sum_{i=1}^{N} w_t^{(i)} \, \delta_{x_t^{(i)}}(x), \qquad \sum_{i=1}^{N} w_t^{(i)} = 1$$

The algorithm applies **Sampling Importance Resampling (SIR)** to propagate states, update weights via Bayesian likelihoods, and systematically resample to avoid particle degeneracy.

---

## 🔬 Experiments & Results

### 🌊 1D Tracking: The Ornstein-Uhlenbeck Process
We simulate an autonomous underwater vehicle experiencing mean-reverting friction and random current impacts. Using the 1D Kalman-Bucy filter, the model successfully tracks the true state over time from constant white-noise sonar readings.

### 🚀 Multi-dimensional System: 3D Missile Tracking
A continuous-time multi-dimensional Kalman-Bucy filter tracks a ballistic missile in a 3D space. 
- **State**: 6D vector (3D position + 3D velocity).
- **Observations**: A ground-based radar providing noisy 3D position measurements only.
The filter optimally infers the unobserved velocities and smooths the trajectory through the `solve_ivp` integration of the Riccati equation.

### 🤖 Non-Linear Application: Monte Carlo Localization (MCL)
A differential-drive robot operates in a 2D plane with known landmarks but absolute global uncertainty. 
- **Dynamics**: Non-linear unicycle kinematics with Brownian slip noise.
- **Observations**: Highly non-linear LiDAR range readings yielding multimodal posteriors.
A Particle Filter with 3,000 particles is implemented. The initial uniform cloud quickly converges into a precise estimation point, handling non-linearities where classical Kalman filters would fail.

---

## ⚖️ Strengths & Limitations

| Model | ✅ Strengths | ⚠️ Limitations |
|:---|:---|:---|
| **Kalman-Bucy Filter** | **Mathematically exact** closed-form solution. Extremely computationally efficient. | **Strictly limited** to linear dynamics and Gaussian noise. |
| **Particle Filters (SMC)** | Handles **highly non-linear** kinematics and multimodal/non-Gaussian distributions effortlessly. | **Curse of dimensionality**: requires exponentially more particles in high dimensions. Prone to **particle degeneracy**. |

---

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![LaTeX](https://img.shields.io/badge/latex-%23008080.svg?style=for-the-badge&logo=latex&logoColor=white)

</div>

**Main libraries:** `numpy` · `scipy.integrate` (`solve_ivp`) · `scipy.interpolate` · `matplotlib`
