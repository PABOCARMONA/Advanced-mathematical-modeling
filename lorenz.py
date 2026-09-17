"""
Datos del atractor de Lorenz para Neural ODEs.

Cada muestra del dataset es un par (estado inicial, estados posteriores):

    y0    -> tensor (3,)          estado en t = 0
    y     -> tensor (T_i, 3)      estados en tiempos posteriores
    t     -> tensor (T_i,)        esos tiempos, con t[0] > 0 y t creciente

El numero de tiempos T_i y el tiempo final t[-1] son distintos en cada
muestra (muestreo irregular), asi que el batch se construye con padding y
una mascara mediante `collate_variable_length`.
"""

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================
# 1. SIMULACION DEL SISTEMA DE LORENZ
# ============================================

def lorenz_vector_field(y, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Campo vectorial de Lorenz. y: (..., 3) -> (..., 3)."""
    x, yy, z = y[..., 0], y[..., 1], y[..., 2]
    dx = sigma * (yy - x)
    dy = x * (rho - z) - yy
    dz = x * yy - beta * z
    return torch.stack([dx, dy, dz], dim=-1)


def simulate_lorenz(y0, n_steps, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """
    Integra Lorenz con Runge-Kutta 4 sobre una malla temporal uniforme.

    y0      : tensor (3,) o (B, 3) con las condiciones iniciales
    devuelve: traj (n_steps + 1, 3) o (B, n_steps + 1, 3), y ts (n_steps + 1,)
    """
    y = y0.clone()
    traj = [y]
    for _ in range(n_steps):
        k1 = lorenz_vector_field(y, sigma, rho, beta)
        k2 = lorenz_vector_field(y + 0.5 * dt * k1, sigma, rho, beta)
        k3 = lorenz_vector_field(y + 0.5 * dt * k2, sigma, rho, beta)
        k4 = lorenz_vector_field(y + dt * k3, sigma, rho, beta)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(y)
    ts = torch.arange(n_steps + 1, dtype=y0.dtype) * dt
    return torch.stack(traj, dim=-2), ts


def generate_lorenz_trajectories(n_traj=20, n_steps=5000, dt=0.01,
                                 transient_steps=1000, spread=20.0, seed=0):
    """
    Genera `n_traj` trayectorias largas sobre el atractor.

    Se descarta un transitorio inicial para que los puntos esten ya sobre el
    atractor. Devuelve un tensor (n_traj, n_steps + 1, 3) y el paso dt.
    """
    g = torch.Generator().manual_seed(seed)
    y0 = (torch.rand(n_traj, 3, generator=g) - 0.5) * 2 * spread
    y0 = y0 + torch.tensor([0.0, 0.0, 25.0])          # centrado en el atractor

    if transient_steps > 0:
        burn, _ = simulate_lorenz(y0, transient_steps, dt)
        y0 = burn[:, -1, :]

    traj, _ = simulate_lorenz(y0, n_steps, dt)
    return traj, dt


# ============================================
# 2. DATASET DE VENTANAS DE LONGITUD VARIABLE
# ============================================

class LorenzWindowDataset(Dataset):
    """
    Dataset de pares (estado inicial, estados futuros).

    Sobre unas trayectorias de referencia se recortan ventanas que empiezan en
    un instante aleatorio. Para cada ventana se sortea:

      * el horizonte H (numero de pasos de la malla fina hasta el ultimo
        tiempo observado) en [min_horizon, max_horizon]  -> tiempos finales
        distintos,
      * la longitud T (numero de observaciones dentro de ese horizonte) en
        [min_len, max_len]                                -> longitudes
        distintas.

    Los T tiempos se eligen sin reemplazo dentro del horizonte y se ordenan,
    forzando que el ultimo sea exactamente H para que el tiempo final sea el
    sorteado. Asi dos muestras con la misma longitud pueden acabar en tiempos
    diferentes, y dos muestras con el mismo tiempo final pueden tener
    longitudes diferentes.

    Parametros
    ----------
    trajectories : (n_traj, L, 3) trayectorias de referencia
    dt           : paso de la malla fina de `trajectories`
    n_samples    : numero de ventanas del dataset
    min_len/max_len         : numero de observaciones por ventana
    min_horizon/max_horizon : horizonte en pasos de la malla fina
    regular_grid : si True, los tiempos son equiespaciados dentro del
                   horizonte; si False (por defecto), muestreo irregular
    seed         : semilla para el sorteo de las ventanas
    """

    def __init__(self, trajectories, dt, n_samples=2000,
                 min_len=10, max_len=50,
                 min_horizon=50, max_horizon=300,
                 regular_grid=False, seed=0):
        if max_len > min_horizon:
            raise ValueError("min_horizon debe ser >= max_len para poder "
                             "elegir tiempos distintos dentro del horizonte")

        self.traj = trajectories
        self.dt = float(dt)
        self.n_samples = int(n_samples)
        self.min_len, self.max_len = int(min_len), int(max_len)
        self.min_horizon, self.max_horizon = int(min_horizon), int(max_horizon)
        self.regular_grid = regular_grid

        n_traj, length, _ = trajectories.shape
        g = torch.Generator().manual_seed(seed)

        # Se sortean todas las ventanas por adelantado: el dataset es
        # determinista y se repite igual en cada epoca.
        self.traj_idx = torch.randint(n_traj, (self.n_samples,), generator=g)
        self.horizons = torch.randint(self.min_horizon, self.max_horizon + 1,
                                      (self.n_samples,), generator=g)
        self.lengths = torch.randint(self.min_len, self.max_len + 1,
                                     (self.n_samples,), generator=g)
        max_start = length - 1 - self.max_horizon
        if max_start <= 0:
            raise ValueError("las trayectorias son demasiado cortas para "
                             "max_horizon")
        self.starts = torch.randint(max_start, (self.n_samples,), generator=g)

        # Offsets (en pasos de la malla fina) de cada observacion.
        self.offsets = [self._sample_offsets(int(self.horizons[i]),
                                             int(self.lengths[i]), g)
                        for i in range(self.n_samples)]

    def _sample_offsets(self, horizon, n_obs, generator):
        """Offsets crecientes en [1, horizon], el ultimo igual a horizon."""
        if self.regular_grid:
            off = torch.linspace(horizon / n_obs, horizon, n_obs)
            return off.round().long().clamp(min=1)
        # n_obs - 1 offsets distintos en [1, horizon - 1], mas el horizonte.
        perm = torch.randperm(horizon - 1, generator=generator)[:n_obs - 1] + 1
        return torch.cat([perm.sort().values, torch.tensor([horizon])]).long()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        traj = self.traj[int(self.traj_idx[idx])]
        start = int(self.starts[idx])
        offsets = self.offsets[idx]

        y0 = traj[start]                      # (3,)
        y = traj[start + offsets]             # (T_i, 3)
        t = offsets.to(y0.dtype) * self.dt    # (T_i,)
        return y0, t, y


# ============================================
# 3. COLLATE CON PADDING + MASCARA
# ============================================

def collate_variable_length(batch):
    """
    Agrupa muestras de longitud distinta rellenando hasta la mas larga.

    Devuelve
    --------
    y0   : (B, 3)          estados iniciales
    t    : (B, T_max)      tiempos (relleno con 0)
    y    : (B, T_max, 3)   estados posteriores (relleno con 0)
    mask : (B, T_max) bool True en las posiciones observadas
    """
    y0 = torch.stack([b[0] for b in batch])
    lengths = [b[1].numel() for b in batch]
    t_max = max(lengths)

    t = torch.zeros(len(batch), t_max, dtype=y0.dtype)
    y = torch.zeros(len(batch), t_max, 3, dtype=y0.dtype)
    mask = torch.zeros(len(batch), t_max, dtype=torch.bool)
    for i, (_, ti, yi) in enumerate(batch):
        n = ti.numel()
        t[i, :n], y[i, :n], mask[i, :n] = ti, yi, True
    return y0, t, y, mask


def make_lorenz_dataloader(batch_size=32, shuffle=True, num_workers=0,
                           n_traj=20, n_steps=5000, dt=0.01, seed=0,
                           **dataset_kwargs):
    """Atajo: genera las trayectorias, el dataset y el DataLoader."""
    traj, dt = generate_lorenz_trajectories(n_traj=n_traj, n_steps=n_steps,
                                            dt=dt, seed=seed)
    dataset = LorenzWindowDataset(traj, dt, seed=seed, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=collate_variable_length)
    return dataset, loader


# ============================================
# 4. COMPROBACION RAPIDA
# ============================================

if __name__ == "__main__":
    torch.manual_seed(42)

    dataset, loader = make_lorenz_dataloader(batch_size=8, n_samples=500)

    print(f"muestras: {len(dataset)}")
    for i in range(3):
        y0, t, y = dataset[i]
        print(f"  muestra {i}: y0 {tuple(y0.shape)}, y {tuple(y.shape)}, "
              f"t final = {t[-1]:.3f}")

    y0, t, y, mask = next(iter(loader))
    print("\nbatch:")
    print(f"  y0   {tuple(y0.shape)}")
    print(f"  t    {tuple(t.shape)}")
    print(f"  y    {tuple(y.shape)}")
    print(f"  mask {tuple(mask.shape)}, observaciones por muestra: "
          f"{mask.sum(1).tolist()}")
    print("  tiempos finales: "
          f"{[round(float(ti[m][-1]), 3) for ti, m in zip(t, mask)]}")
