import plotly.graph_objects as go
import os
from scipy.optimize import minimize
import math
from math import ceil, floor
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

USE_GPU = os.getenv("USE_GPU", "0") == "1"

if USE_GPU:
    try:
        import cupy as xp
        import cupyx.scipy.fft as xfft
        on_gpu = True

        asnumpy = xp.asnumpy    # CuPy → NumPy
        asarray = xp.asarray   # Python/NumPy → CuPy
    except ImportError:
        # fallback if cupy not installed
        import numpy as xp
        import scipy.fft as xfft
        on_gpu = False

        asnumpy = lambda a: a
        asarray = xp.asarray
else:
    import numpy as xp
    import scipy.fft as xfft
    on_gpu = False

    asnumpy = lambda a: a
    asarray = xp.asarray

# -------- light helpers (backend-agnostic) --------
def wrap(x):
    return (x + xp.pi) % (2 * xp.pi) - xp.pi


def to_device(x):
    """Move/convert Python or NumPy data to the active backend array (CuPy on GPU, NumPy on CPU)."""
    return asarray(x)

def to_host(x):
    """Ensure a NumPy array on host memory (identity on CPU)."""
    return asnumpy(x)

def to_scalar(x):
    """Return a Python scalar if x is 0-d array/array-like; otherwise return x unchanged."""
    # Works for both NumPy and CuPy
    try:
        if getattr(x, "shape", None) == () or getattr(x, "shape", None) == (1,):
            return x.item()
    except Exception:
        pass
    return x

def sqlist(lst):
    return xp.array([to_scalar(item) for item in lst])

def myfft(chirp_data, n, plan):
    if USE_GPU:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex128), n=n, plan=plan))
    else:
        return xfft.fftshift(xfft.fft(chirp_data.astype(xp.complex128), n=n))


def optimize_1dfreq_fast(sig2, fs, freq1, margin):
    # minimize converts values to numpy arrays for freq, print(type(freq), type(fs), type(ydata)) returns <class 'numpy.ndarray'> <class 'float'> <class 'cupy.ndarray'>
    def obj1(freq, fs, ydata):
        return to_scalar(-xp.abs(ydata.dot(xp.exp(xp.arange(ydata.shape[0]) / fs * -1j * 2 * xp.pi * to_device(freq)))))
    result = minimize(obj1, to_scalar(freq1), args=(fs, sig2), bounds=[(freq1 - margin, freq1 + margin)]) #!!!
    return result.x[0], - result.fun / xp.sum(xp.abs(sig2))


def around(x):
    """Round to nearest integer (works for Python num or 0-d array)."""
    return round(float(to_scalar(x)))

def optimize_1dfreq(sig2, tsymbr, freq, margin):
    def obj1(xdata, ydata, freq):
        return xp.abs(ydata.dot(xp.exp(-1j * 2 * xp.pi * freq * xdata)))

    margin = 500
    val = obj1(tsymbr, sig2, freq)
    for i in range(10):
        xvals = xp.linspace(freq - margin, freq + margin, 1001)
        yvals = [obj1(tsymbr, sig2, f) for f in xvals]
        yvals = sqlist(yvals)
        freq = xvals[xp.argmax(yvals)]
        valnew = xp.max(yvals)
        if valnew < val * (1 - 1e-7):
            pltfig1(xvals, yvals, addvline=(freq,), title=f"Optimize_1dfreq {i=} {val=} {valnew=}").show()
        assert valnew >= val * (1 - 1e-7), f"{val=} {valnew=} {i=} {val-valnew=}"
        if abs(valnew - val) < 1e-7: margin /= 4
        val = valnew
    return freq, val / xp.sum(xp.abs(sig2))

def pltfig(datas, title = None, yaxisrange = None, modes = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    datas : list of tuples
        Each tuple contains two lists or array-like elements, the data for the x and y axes.
        If only y data is provided, x data will be generated using xp.arange.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if not all(len(data) == 2 for data in datas): datas = [(xp.arange(len(data)), data) for data in datas]
    # if not(all(len(data) == 2 for data in datas) and all(type(to_device(data[0])) == xp.array and type(to_device(data[1])) == xp.array  for data in datas)):
    #     print(len(datas[0]))
    #     print(len(datas[1]))
    #     print(type(datas[0][0]))
    #     print(type(datas[0][1]))
    #     print(type(datas[1][0]))
    #     print(type(datas[1][1]))
    #     print(datas[0][0].dtype)
    #     print(datas[0][1].dtype)
    #     print(datas[1][0].dtype)
    #     print(datas[1][1].dtype)
    assert all(len(data) == 2 for data in datas) 
    assert all(isinstance(to_device(data[0]), xp.ndarray) and isinstance(to_device(data[1]), xp.ndarray) for data in datas)
    if modes is None:
        modes = ['lines' for _ in datas]
    elif isinstance(modes, str):
        modes = [modes for _ in datas]
    for idx, ((xdata, ydata), mode) in enumerate(zip(datas, modes)):
        if line == None and idx == 1: line = dict(dash='dash')
        fig.add_trace(go.Scatter(x=to_host(xdata), y=to_host(ydata), mode=mode, marker=marker, line=line))
        assert len(to_host(xdata)) == len(to_host(ydata))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig



def pltfig1(xdata, ydata, title = None, yaxisrange = None, mode = None, marker = None, addvline = None, addhline = None, line_dash = None, fig = None, line=None):
    """
    Plot a figure with the given data and parameters.

    Parameters:
    xdata : list or array-like or None
        The data for the x-axis.
        If is None, and only y data is provided, x data will be generated using xp.arange.
    ydata : list or array-like
        The data for the y-axis.
    title : str, optional
        The title of the plot (default is None).
    yaxisrange : tuple, optional
        The range for the y-axis as a tuple (min, max) (default is None).
    mode : str, optional
        The mode of the plot (e.g., 'line', 'scatter') (default is None).
    marker : str, optional
        The marker style for the plot (default is None).
    addvline : float, optional
        The x-coordinate for a vertical line (default is None).
    addhline : float, optional
        The y-coordinate for a horizontal line (default is None).
    line_dash : str, optional
        The dash style for the line (default is None).
    fig : matplotlib.figure.Figure, optional
        The figure object to plot on (default is None).
    line : matplotlib.lines.Line2D, optional
        The line object for the plot (default is None).

    Returns:
    None
    """
    if xdata is None: xdata = xp.arange(len(ydata))
    if fig is None: fig = go.Figure(layout_title_text=title)
    elif title is not None: fig.update_layout(title_text=title)
    if mode is None: mode = 'lines'
    fig.add_trace(go.Scatter(x=to_host(xdata), y=to_host(ydata), mode=mode, marker=marker, line=line))
    assert len(to_host(xdata)) == len(to_host(ydata))
    pltfig_hind(addhline, addvline, line_dash, fig, yaxisrange)
    return fig

def pltfig_hind(addhline, addvline, line_dash_in, fig, yaxisrange):
    if yaxisrange: fig.update_layout(yaxis=dict(range=yaxisrange), )
    if addvline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addvline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addvline))]
        else: line_dash = line_dash_in
        for x, ldash in zip(addvline, line_dash): fig.add_vline(x=to_scalar(x), line_dash=ldash)
    if addhline is not None:
        if line_dash_in is None:
            line_dash = ['dash' for _ in range(len(addhline))]
            line_dash[0] = 'dot'
        elif isinstance(line_dash_in, str):
            line_dash = [line_dash_in for _ in range(len(addhline))]
        else: line_dash = line_dash_in
        for y, ldash in zip(addhline, line_dash): fig.add_hline(y=to_scalar(y), line_dash=ldash)
