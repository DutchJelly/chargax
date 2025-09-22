from typing import Optional, TypedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scipy.signal import savgol_filter
from scipy import interpolate


class EpisodeReturn(TypedDict):
    episode_return: float
    env_step: int


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Environment step")
        self.ax.set_ylabel("Episode Return")
        if title is not None:
            self.ax.set_title(title)
            
        self._last_color = None
        self._custom_labels = []

    def add_curve(self, x, y, label=None, std_dev=None):
        """y: vector of average reward results
        label: string to appear as label in plot legend"""
        if label is not None:
            (line,) = self.ax.plot(x, y, label=label)
        else:
            (line,) = self.ax.plot(x, y)

        if std_dev is not None:
            self.ax.fill_between(
                x, y - std_dev, y + std_dev, color=line.get_color(), alpha=0.2
            )
        
        self._last_color = line.get_color()
        
    def add_custom_label(self, **args):
        self._custom_labels.append(Line2D([0], [0], **args))

    def show_legend(self, **args):
        lines = []
        lines.extend(filter(lambda x: "_child" not in str(x.get_label()), self.ax.get_lines()))
        lines.extend(self._custom_labels)
        self.ax.legend(handles=lines, **args)

    def set_ylim(self, lower: float, upper: float):
        self.ax.set_ylim(bottom=lower, top=upper)

    def add_hline(self, height, label: Optional[str] = None, use_last_color=False, structure="--"):
        if use_last_color:
            assert self._last_color is not None, "can only use last color if there's a previous line"
            
            self.ax.axhline(height, ls=structure, c=self._last_color)
        else:            
            self.ax.axhline(height, ls=structure, c="k", label=label)

    def save(self, name="test.png", **legend_args):
        """name: string for filename of saved figure"""
        self.ax.legend(**legend_args)
        self.fig.savefig(name, dpi=300)


def average_over_repetition(repetitions: list[list[EpisodeReturn]]):
    Y = []

    x = np.array([ep_res["env_step"] for ep_res in repetitions[0]])

    for results in repetitions:
        y = np.array([ep_res["episode_return"] for ep_res in results])
        Y.append(y)

    Y = np.array(Y)

    return x, Y.mean(axis=0), Y.std(axis=0)


def interpolate_and_average_over_repetitions(
    repetitions: list[list[EpisodeReturn]], step_size: int
):
    all_env_steps = [
        ep_res["env_step"] for results in repetitions for ep_res in results
    ]
    min_step = min(all_env_steps)
    max_step = max(all_env_steps)

    X = np.arange(min_step, max_step + step_size, step_size)

    repetitions_interpolated = []

    for results in repetitions:
        x = np.array([ep_res["env_step"] for ep_res in results])
        y = np.array([ep_res["episode_return"] for ep_res in results])

        interpolator = interpolate.interp1d(
            x, y, kind="linear", bounds_error=False, fill_value=(y[0], y[-1])
        )

        y_interpolated = np.array(interpolator(X))
        repetitions_interpolated.append(y_interpolated)

    return X, np.array(repetitions_interpolated).mean(axis=0)


def smooth(y, window, poly=2):
    """
    y: vector to be smoothed
    window: size of the smoothing window"""
    if len(y) == 0:
        raise RuntimeError("Length of y cannot be 0.")
    if window > len(y):
        print(f"Window size ({window}) is larger than data length ({len(y)})")
        window = len(y)
    if poly >= window:
        print(f"Poly ({poly}) is larger or equal to window length of {window}")
        poly = window - 1
    return savgol_filter(y, window, poly)
