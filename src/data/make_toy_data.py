from typing import Tuple
from warnings import warn
import numpy as np


def random_unit_box(n_samples: int) -> np.ndarray:

    x = np.random.uniform(size=n_samples)
    y = np.random.uniform(size=n_samples)

    return np.array(list(zip(x, y)))


def linear(n_samples: int,
           noise_amplitude: float = None,
           slope: float = 1,
           homoscedastic_noise: bool = True) -> np.ndarray:
    
    if homoscedastic_noise:
        x = np.random.uniform(size=n_samples)
        y_0 = slope * x

        if noise_amplitude is not None:
            y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
        else:
            y = y_0

        return np.array(list(zip(x, y, y_0)))
    else:
        data_0 = linear(n_samples, slope=slope)

        noise_amplitude = 0.25 if noise_amplitude is None else noise_amplitude
        data_0 = add_heteroscedastic_noise(data_0, noise_amplitude=noise_amplitude)

        return data_0


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


def ellipse(n_samples: int,
            noise_amplitude: float = None) -> np.ndarray:
    
    X_AXIS, Y_AXIS = 3, 1

    t = np.random.uniform(high=2 * np.pi, size=n_samples)
    x_0 = X_AXIS * np.cos(t)
    y_0 = Y_AXIS * np.sin(t)

    if noise_amplitude is not None:
        x = x_0 + 3 * (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
        y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples)) - noise_amplitude / 2
    else:
        x = x_0
        y = y_0

    return np.array(list(zip(x, y, x_0, y_0)))


def two_lines(n_samples: int,
              noise_amplitude: float = None,
              slopes: Tuple[int] = (3, 0.5),
              homoscedastic_noise: bool = True) -> np.ndarray:

    if homoscedastic_noise:
        line_0 = linear(n_samples=n_samples//2, noise_amplitude=noise_amplitude, slope=slopes[0])
        line_1 = linear(n_samples=n_samples//2, noise_amplitude=noise_amplitude, slope=slopes[1])

        return np.vstack((line_0, line_1))
    else:
        data_0 = two_lines(n_samples, slopes=slopes)

        noise_amplitude = 0.25 if noise_amplitude is None else noise_amplitude
        data_0 = add_heteroscedastic_noise(data_0, noise_amplitude=noise_amplitude)

        return data_0


def line_and_parabola_down(n_samples: int,
                           noise_amplitude: float = None,
                           homoscedastic_noise: bool = True) -> np.ndarray:
    
    if homoscedastic_noise:
        line_0 = linear(n_samples=n_samples//2, noise_amplitude=noise_amplitude, slope=2)

        x = np.random.uniform(size=n_samples//2)
        y_0 = 1/2 - 2 * (x.copy() - 1/2)**2

        if noise_amplitude is not None:
            y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples//2)) - noise_amplitude / 2
        else:
            y = y_0

        return np.vstack((line_0, np.array(list(zip(x, y, y_0)))))
    else:
        data_0 = line_and_parabola_down(n_samples)

        noise_amplitude = 0.25 if noise_amplitude is None else noise_amplitude
        data_0 = add_heteroscedastic_noise(data_0, noise_amplitude=noise_amplitude)

        return data_0


def line_and_parabola_up(n_samples: int,
                         noise_amplitude: float = None,
                         homoscedastic_noise: bool = True) -> np.ndarray:
    
    if homoscedastic_noise:
        line_0 = linear(n_samples=n_samples//2, noise_amplitude=noise_amplitude, slope=2)

        x = np.random.uniform(size=n_samples//2)
        y_0 =  1 * x.copy()**2

        if noise_amplitude is not None:
            y = y_0 + (noise_amplitude * np.random.uniform(size=n_samples//2)) - noise_amplitude / 2
        else:
            y = y_0

        return np.vstack((line_0, np.array(list(zip(x, y, y_0)))))
    else:
        data_0 = line_and_parabola_up(n_samples)

        noise_amplitude = 0.25 if noise_amplitude is None else noise_amplitude
        data_0 = add_heteroscedastic_noise(data_0, noise_amplitude=noise_amplitude)

        return data_0


def non_coexistence(n_samples: int,
                    noise_amplitude: float = None) -> np.ndarray:
    
    line_0 = linear(n_samples=n_samples//2, noise_amplitude=None, slope=0)
    line_1 = linear(n_samples=n_samples//2, noise_amplitude=None, slope=0)

    if noise_amplitude is not None:
        line_0[:, 1] = line_0[:, 1] + np.abs(noise_amplitude * np.random.uniform(size=n_samples//2)) # - noise_amplitude / 2
        line_1[:, 1] = line_1[:, 1] + np.abs(noise_amplitude * np.random.uniform(size=n_samples//2)) # - noise_amplitude / 2
    # else:
    #     line_0 = line_0_0
    #     line_1 = line_1_0
    
    line_1_flipped = line_1.copy()
    line_1_flipped[:, 0] = line_1[:, 1]
    line_1_flipped[:, 1] = line_1[:, 0]
    line_1_flipped[:, 2] = line_1[:, 0]
    
    # line_0 = np.delete(line_0, 2, 1)
    # line_1_flipped = np.delete(line_1_flipped, 2, 1)

    return np.vstack((line_0, line_1_flipped))


def add_heteroscedastic_noise(data: np.ndarray,
                              noise_amplitude: float) -> np.ndarray:

    return_data = data.copy()

    for i, (x, y, _) in enumerate(return_data):

        if y < 0:
            warn(f"Found y value below 0, this is not expected and may cause issues with the while loop below...")
        
        new_y = -1
        while new_y < 0:
            new_y = y + ((0.05 + x**2) * np.random.normal(scale=noise_amplitude))
        
        return_data[i, 1] = new_y

    return return_data
