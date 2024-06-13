import numpy as np


def random_unit_box(n_samples: int) -> np.ndarray:

    x = np.random.uniform(size=n_samples)
    y = np.random.uniform(size=n_samples)

    return np.array(list(zip(x, y)))


def linear(n_samples: int,
           noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = x

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def parabolic(n_samples: int,
              noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(low=-1/2, high=1/2, size=n_samples)
    y_0 = 4 * x.copy()**2

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def cubic(n_samples: int,
          noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(low=-1.3, high=1.1, size=n_samples)
    y_0 = 4 * x.copy()**3 + x.copy()**2 - 4 * x.copy()

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def exponential(n_samples: int,
                noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = 10**(10 * x.copy()) - 1

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def linear_periodic_med_freq(n_samples: int,
                             noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = np.sin(10 * np.pi * x.copy()) + x.copy()

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def sinus_fourier(n_samples: int,
                  noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = np.sin(16 * np.pi * x.copy())

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def sinus_non_fourier(n_samples: int,
                      noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = np.sin(13 * np.pi * x.copy())

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def sinus_varying(n_samples: int,
                  noise_amplitude: float = None) -> np.ndarray:
    
    x = np.random.uniform(size=n_samples)
    y_0 = np.sin(7 * np.pi * x.copy() * (1 + x.copy()))

    if noise_amplitude is not None:
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        y = y_0

    return np.array(list(zip(x, y, y_0)))


def categorical(n_samples: int) -> np.ndarray:

    choices = [[1, 0.287], [2, 0.796], [3, 0.290], [4, 0.924], [5, 0.717]]
    
    rng = np.random.default_rng()
    
    return rng.choice(a=choices, size=n_samples, replace=True)
