import matplotlib.pyplot as plt
import itertools


def plot_nonempty(array, name, axes, color='k'):
    marker = itertools.cycle(('+', 'o', '.', 'x', '*'))
    style = itertools.cycle(('-', '--', '-', '--'))

    if not axes:
        fig, axes = plt.subplots(1, 1)

    legend = []

    n_t_levels = array.shape[1]
    n_c_levels = array.shape[2]
    for i in range(n_t_levels):
        for j in range(n_c_levels):
            sweep = array[:, i, j].dropna('fd')
            if sweep.shape[0] != 0:
                legend.append(name + ' ' + str(sweep.t_levels.values[()]) + 'x' + str(sweep.c_levels.values[()]))
                sweep.plot.line(color=color, marker=marker.next(), ax=axes, ls=style.next(), lw=3, markersize=10)

    return legend


def multi_series_plotter(x_dimension, cut_dimension, series, ax=False, linestyle='-'):
    if ax is False:
        fig, ax = plt.subplots(1, 1)

    for cut_value, cut in series.groupby(cut_dimension):
        x_values = cut.index.get_level_values(x_dimension)
        ax.plot(x_values, cut, linestyle=linestyle)


def hilbert_calibration_plotter(results):
    import itertools
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    style = itertools.cycle(('-', '-', '--', '--'))

    fig, axes = plt.subplots(1, 1)

    legend = []

    n_t_levels = results.shape[1]
    n_c_levels = results.shape[2]
    for i in range(n_t_levels):
        for j in range(n_c_levels):
            sweep = results[:, i, j].dropna('fd')
            if sweep.shape[0] != 0:
                legend.append(str(sweep.t_levels.values[()]) + 'x' + str(sweep.c_levels.values[()]))
                sweep.plot.line(marker=marker.next(), ax=axes, ls=style.next(), lw=5)

    axes.legend(legend)

    plt.show()