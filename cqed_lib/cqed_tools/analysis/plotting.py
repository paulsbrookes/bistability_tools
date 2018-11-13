import matplotlib.pyplot as plt
import itertools
import numpy as np


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


def hilbert_calibration_plotter(results, axes=None):
    import itertools
    marker = itertools.cycle((',', '+', '.', 'o', '*'))
    style = itertools.cycle(('-', '-', '--', '--'))

    if axes is None:
        print('Axes not given.')
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


def plot_time_constants_sim(time_constants, axes=None, ls='--', marker='o', markersize=10, lower_bound=0,
                            interpolate=True):
    time_constants = time_constants.replace([np.inf, -np.inf], np.nan).dropna()

    mi = time_constants.index
    eps_index = time_constants.index.names.index('eps')
    eps_values = mi.levels[eps_index]
    eps_values = eps_values[lower_bound:eps_values.shape[0]]

    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    c_iterator = iter(colors)

    for i, eps in enumerate(eps_values):
        color = c_iterator.next()
        # color='r'
        cut = time_constants.xs(eps, level=eps_index)
        cut = cut.sort_index(level='fd')
        fd_array = cut.index.get_level_values('fd')
        cut = cut.astype(np.complex)
        if interpolate:
            func = interp1d(fd_array, cut, kind='cubic')
            fd_points = np.linspace(fd_array[0], fd_array[-1], 1001)
            axes.plot(fd_points, func(fd_points), ls=ls, marker=None, color=color)
            # axes.plot(fd_array, cut.real, ls='', marker=marker, markersize=markersize, color=color)
        else:
            axes.plot(fd_array, cut.real, ls=ls, marker=marker, markersize=markersize, color=color)

    if interpolate:
        c_iterator = iter(colors)
        for i, eps in enumerate(eps_values):
            color = c_iterator.next()
            # color='r'
            cut = time_constants.xs(eps, level=eps_index)
            cut = cut.sort_index(level='fd')
            fd_array = cut.index.get_level_values('fd')
            cut = cut.astype(np.complex)
            axes.plot(fd_array, cut.real, ls='', marker=marker, markersize=markersize, color=color)

    plt.savefig('sim_time_constants.png')

    plt.gca().set_prop_cycle(None)

    return axes


def plot_time_constants_exp(combined, combined_errors, axes=None, loc=0, fmt='-o', ls='-', marker='o', show_errors=True,
                        markersize=12, markeredgewidth=3):

    if axes is None:
        matplotlib.rcParams['figure.figsize'] = (12, 8)

        font = {'weight': 'normal',
                'size': 22}

        matplotlib.rc('font', **font)

        fig, axes = plt.subplots(1, 1)

    n_runs = combined.shape[0]

    for i in np.flip(np.arange(n_runs), axis=0):
        pruned_constants = combined.iloc[i].dropna()
        pruned_errors = combined_errors.iloc[i].dropna()
        # combined.iloc[i].dropna().plot(ax=axes)
        # axes.scatter(combined.iloc[i].index, combined.iloc[i])
        if show_errors:
            axes.errorbar(pruned_constants.index, pruned_constants, yerr=pruned_errors, fmt=fmt)
        else:
            axes.plot(pruned_constants.index, pruned_constants, marker=marker, ls=ls, markersize=markersize,
                      markeredgewidth=markeredgewidth)
    axes.legend(np.flip(combined.index, axis=0), loc=loc, title='Power (dBm)')
    axes.set_xlabel('Drive frequency / GHz')
    axes.set_ylabel(r'Time constant / $\mu$s')
    plt.gca().set_prop_cycle(None)

    return axes