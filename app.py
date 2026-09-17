import torch
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Usando {device}")

import io
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from sklearn.preprocessing import LabelEncoder
except ModuleNotFoundError:
    class LabelEncoder:
        """Equivalente minimo al de sklearn, por si no esta instalado."""

        def fit_transform(self, y):
            self.classes_, codigos = np.unique(y, return_inverse=True)
            return codigos

        def transform(self, y):
            return np.searchsorted(self.classes_, y)

        def inverse_transform(self, codigos):
            return self.classes_[np.asarray(codigos)]

# En un script existe __file__; en un notebook no, ahi se usa el directorio actual.
_BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RUTA_ZIP = _BASE / "character_trajectories" / "character_trajectories.zip"


def cargar_datos(ruta_zip=RUTA_ZIP):
    """Lee el .mat directamente desde el zip local, sin descargar nada.

    Devuelve:
        trajectories:  lista de 2858 arrays (T_i, 3) con columnas [t, x, y];
                       x e y son posiciones (velocidades integradas) escaladas a [-1, 1]
        labels:        array (N,) con el codigo entero de cada caracter
        label_encoder: LabelEncoder ajustado, label_encoder.classes_ da las 20 letras
    """
    with zipfile.ZipFile(ruta_zip) as z:
        nombre = next(n for n in z.namelist() if n.endswith(".mat"))
        mat = loadmat(io.BytesIO(z.read(nombre)))

    mixout = mat["mixout"][0]                       # 2858 secuencias (3, T)
    consts = mat["consts"][0, 0]
    key = np.array([str(k[0]) for k in consts["key"][0]])
    etiquetas = key[consts["charlabels"][0] - 1]    # indices MATLAB: 1-based
    dt = float(consts["dt"][0, 0])                  # 0.005 s

    trajectories = []
    for seq in mixout:
        x = np.cumsum(seq[0]) * dt                  # integrar velocidades -> posicion
        y = np.cumsum(seq[1]) * dt
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 - 1
        y = (y - y.min()) / (y.max() - y.min() + 1e-8) * 2 - 1
        t = np.arange(len(x)) * dt
        trajectories.append(np.column_stack([t, x, y]))

    le = LabelEncoder()
    encoded = le.fit_transform(etiquetas)
    print(f"Total de trayectorias: {len(trajectories)}")
    print(f"Clases ({len(le.classes_)}): {le.classes_}")
    return trajectories, encoded, le


def a_tensores(trajectories, labels, device=device):
    """Rellena las secuencias a la longitud maxima y las pasa a torch.

    Devuelve x (N, T_max, 3) con columnas [t, x, y], mascara (N, T_max) con
    True en los pasos reales, longitudes (N,) e y (N,).
    """
    n = len(trajectories)
    longitudes = np.array([len(t) for t in trajectories])
    t_max = int(longitudes.max())

    x = np.zeros((n, t_max, 3), dtype=np.float32)
    mascara = np.zeros((n, t_max), dtype=bool)
    for i, t in enumerate(trajectories):
        x[i, : len(t)] = t
        mascara[i, : len(t)] = True

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(mascara).to(device),
        torch.from_numpy(longitudes).to(device),
        torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device),
    )


if __name__ == "__main__":
    trajectories, labels, label_encoder = cargar_datos()
    x, mascara, longitudes, y = a_tensores(trajectories, labels)

    print(f"x: {tuple(x.shape)}  mascara: {tuple(mascara.shape)}  y: {tuple(y.shape)}")
    print(f"longitud min/media/max: {longitudes.min()} / {longitudes.float().mean():.1f} / {longitudes.max()}")
    print(f"primera muestra: caracter '{label_encoder.classes_[labels[0]]}', {longitudes[0]} pasos")
