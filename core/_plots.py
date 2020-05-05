"""Extension module containing utilities for plotting"""

from typing import Dict, List, Set, Tuple
from typing import Sequence, Iterable, Iterator, Collection
from typing import Any, Optional, Type, Union, IO

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def pie(self, ax=None, pie_props=None):
    size = 0.2
    radius = 0.22

    if ax is None:
        ax = plt.gca()

    def getpieces(c, pieces=None, level=0, ref="0"):
        if not pieces:
            pieces = {}
        if level not in pieces:
            pieces[level] = {}

        if c._clusterdict:
            ring = {k: len(v) for k, v in c._clusterdict.items()}
            ringsum = np.sum(list(ring.values()))
            ring = {k: v/ringsum for k, v in ring.items()}
            pieces[level][ref] = ring

            if c._children:
                for number, child in c._children.items():
                    getpieces(
                        child,
                        pieces=pieces,
                        level=level+1,
                        ref=".".join([ref, str(number)])
                    )

        return pieces
    p = getpieces(self)

    ringvalues = []
    for j in range(np.max(list(p[0]['0'].keys())) + 1):
        if j in p[0]['0']:
            ringvalues.append(p[0]['0'][j])

    ax.pie(
        ringvalues, radius=radius, colors=None,
        wedgeprops=dict(width=size, edgecolor='w')
        )

    # iterating through child levels
    for i in range(1, np.max(list(p.keys()))+1):
        # what has not been reclustered:
        reclustered = np.asarray(
            [key.rsplit('.', 1)[-1] for key in p[i].keys()]
            ).astype(int)
        # iterate over clusters of parent level
        for ref, values in p[i-1].items():
            # account for not reclustered clusters
            for number in values:
                if number not in reclustered:
                    p[i][".".join([ref, str(number)])] = {0: 1}

        # iterate over clusters of child level
        for ref in p[i]:
            preref = ref.rsplit('.', 1)[0]
            sufref = int(ref.rsplit('.', 1)[-1])
            p[i][ref] = {
                k: v*p[i-1][preref][sufref]
                for k, v in p[i][ref].items()
            }

        ringvalues = []
        for ref in sorted(list(p[i].keys())):
            for j in p[i][ref]:
                ringvalues.append(p[i][ref][j])

        ax.pie(
            ringvalues, radius=radius + i*size, colors=None,
            wedgeprops=dict(width=size, edgecolor='w')
            )


def summarize(self, ax=None, quant: str="time", treat_nan=None,
                ax_props=None, contour_props=None) -> Tuple:
    """Generates a 2D plot of property values ("time", "noise",
    "n_clusters", "largest") against cluster parameters
    radius_cutoff and cnn_cutoff."""

    if len(self.summary) == 0:
        raise LookupError(
"""No cluster result summary present"""
            )

    pivot = self.summary.groupby(
        ["radius_cutoff", "cnn_cutoff"]
        ).mean()[quant].reset_index().pivot(
            "radius_cutoff", "cnn_cutoff"
            )


    X_, Y_ = np.meshgrid(
        pivot.index.values, pivot.columns.levels[1].values
        )

    values_ = pivot.values.T

    if treat_nan is not None:
        values_[np.isnan(values_)] == treat_nan

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax_props_defaults = {
        "xlabel": "$R$",
        "ylabel": "$N$",
    }

    if ax_props is not None:
        ax_props_defaults.update(ax_props)

    contour_props_defaults = {
            "cmap": mpl.cm.inferno,
        }

    if contour_props is not None:
        contour_props_defaults.update(contour_props)

    plotted = []

    plotted.append(
        ax.contourf(X_, Y_, values_, **contour_props_defaults)
        )

    ax.set(**ax_props_defaults)

    return fig, ax, plotted


def evaluate(
        self,
        ax: Optional[Type[mpl.axes.SubplotBase]] = None,
        clusters: Optional[List[int]] = None,
        original: bool = False, plot: str = 'dots',
        parts: Optional[Tuple[Optional[int]]] = None,
        points: Optional[Tuple[Optional[int]]] = None,
        dim: Optional[Tuple[int, int]] = None,
        ax_props: Optional[Dict] = None, annotate: bool = True,
        annotate_pos: str = "mean",
        annotate_props: Optional[Dict] = None,
        scatter_props: Optional[Dict] = None,
        scatter_noise_props: Optional[Dict] = None,
        dot_props: Optional[Dict] = None,
        dot_noise_props: Optional[Dict] = None,
        hist_props: Optional[Dict] = None,
        contour_props: Optional[Dict] = None,
        free_energy: bool = True, mask=None,
        threshold=None,
        ):

    """Returns a 2D plot of an original data set or a cluster result

    Args:
        ax: The `Axes` instance to which to add the plot.  If `None`,
        a new `Figure` with `Axes` will be created.

        clusters :
            Cluster numbers to include in the plot.  If `None`,
            consider all.

        original: bool, default=False
            Allows to plot the original data instead of a cluster
            result.  Overrides `clusters`.  Will be considered
            True, if no cluster result is present.

        plot: str, default="dots"
            The kind of plotting method to use.

                * "dots", Use :func:`ax.plot()`

                * "",

        parts: Tuple[int, int, int] (length 3), default=(None, None, None)
            Use a slice (start, stop, stride) on the data parts before
            plotting.

        points: Tuple[int, int, int], default=(None, None, None)
            Use a slice (start, stop, stride) on the data points before
            plotting.

        dim: Tuple[int, int], default=None
            Use these two dimensions for plotting.  If None, uses
            (0, 1).

        annotate: bool, default=True
            If there is a cluster result, plot the cluster numbers.  Uses
            `annotate_pos` to determinte the position of the
            annotations.

        annotate_pos: str or List[Tuple[int, int]], default="mean"
            Where to put the cluster number annotation.  Can be one of:

                * "mean", Use the cluster mean

                * "random", Use a random point of the cluster

            Alternatively a list of x, y positions can be passed to set
            a specific point for each cluster (Not yet implemented)

        annotate_props: Dict, default=None
            Dictionary of keyword arguments passed to
            :func:`ax.annotate(**kwargs)`.

        ax_props: Dict, default=None
            Dictionary of `ax` properties to apply after
            plotting via :func:`ax.set(**ax_props)`.  If None, uses
            defaults that can be also defined in the configuration file.

        (hist, contour, dot, scatter, dot_noise, scatter_noise)_props: Dict, default=None
            Dictionaries of keyword arguments passed to various
            functions.  If None, uses
            defaults that can be also defined in the configuration file.

        mask: Sequence[bool]

    Returns: List of plotted elements
    """

    if dim is None:
        dim = (0, 1)
    elif dim[1] < dim[0]:
        dim = dim[::-1]

    # BROKEN: Fix
    if parts is None:
        parts = (None, None, None)

    if points is None:
        points = (None, None, None)

    # TODO: Avoid copying here
    # _data = [
    #     x[slice(*points), slice(dim[0], dim[1]+1, dim[1]-dim[0])]
    #     for x in _data[slice(*parts)]
    #     ]

    # if mask is not None:
    #     _data = _data[np.asarray(mask)]

    if not original:
        if self.labels is not None:
            if clusters is None:
                clusters = list(self.labels.clusterdict.keys())
        else:
            original = True

    # TODO make this a configuation option
    ax_props_defaults = {
        "xlabel": "$x$",
        "ylabel": "$y$",
    }

    if ax_props is not None:
        ax_props_defaults.update(ax_props)

    annotate_props_defaults = {}

    if annotate_props is not None:
        annotate_props_defaults.update(annotate_props)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # List of drawn objects to return for faster access
    plotted = []

    if plot == 'dots':
        # TODO make this a configuation option
        dot_props_defaults = {
            'lw': 0,
            'marker': '.',
            'markersize': 4,
            'markeredgecolor': 'none',
            }

        if dot_props is not None:
            dot_props_defaults.update(dot_props)

        dot_noise_props_defaults = {
            'color': 'none',
            'lw': 0,
            'marker': '.',
            'markersize': 4,
            'markerfacecolor': 'k',
            'markeredgecolor': 'none',
            'alpha': 0.3
            }

        if dot_noise_props is not None:
            dot_noise_props_defaults.update(dot_noise_props)

        if original:
            # Plot the original data
            plotted.append(ax.plot(
                self.data.data[:, 0],
                self.data.data[:, 1],
                **dot_props_defaults
                ))

        else:
            # Loop through the cluster result
            for cluster, cpoints in self.labels.clusterdict.items():
                cpoints = np.asarray(list(cpoints))

                # plot if cluster is in the list of considered clusters
                if cluster in clusters:
                    # treat noise differently
                    if cluster == 0:
                        plotted.append(ax.plot(
                            self.data.data[cpoints, 0],
                            self.data.data[cpoints, 1],
                            **dot_noise_props_defaults
                            ))

                    else:
                        plotted.append(ax.plot(
                            self.data.data[cpoints, 0],
                            self.data.data[cpoints, 1],
                            **dot_props_defaults
                            ))

                        if annotate:
                            if annotate_pos == "mean":
                                xpos = np.mean(self.data.data[cpoints, 0])
                                ypos = np.mean(self.data.data[cpoints, 1])

                            elif annotate_pos == "random":
                                choosen = random.choice(cpoints)
                                xpos = self.data.data[choosen, 0]
                                ypos = self.data.data[choosen, 1]

                            else:
                                raise ValueError()

                            plotted.append(ax.annotate(
                                f"{cluster}",
                                xy=(xpos, ypos),
                                **annotate_props_defaults
                                ))

    elif plot == 'scatter':

        scatter_props_defaults = {
            's': 10,
        }

        if scatter_props is not None:
            scatter_props_defaults.update(scatter_props)

        scatter_noise_props_defaults = {
            'color': 'k',
            's': 10,
            'alpha': 0.5
        }

        if scatter_noise_props is not None:
            scatter_noise_props_defaults.update(scatter_noise_props)

        if original:
            plotted.append(ax.scatter(
                _data[:, 0],
                _data[:, 1],
                **scatter_props_defaults
                ))

        else:
            for cluster, cpoints in items:
                if cluster in clusters:

                    # treat noise differently
                    if cluster == 0:
                        plotted.append(ax.scatter(
                            _data[cpoints, 0],
                            _data[cpoints, 1],
                            **scatter_noise_props_defaults
                        ))

                    else:
                        plotted.append(ax.scatter(
                            _data[cpoints, 0],
                            _data[cpoints, 1],
                            **scatter_props_defaults
                            ))

                        if annotate:
                            if annotate_pos == "mean":
                                xpos = np.mean(_data[cpoints, 0])
                                ypos = np.mean(_data[cpoints, 1])

                            elif annotate_pos == "random":
                                choosen = random.sample(
                                    cpoints, 1
                                    )
                                xpos = _data[choosen, 0]
                                ypos = _data[choosen, 1]

                            else:
                                raise ValueError()

                            plotted.append(ax.annotate(
                                f"{cluster}",
                                xy=(xpos, ypos),
                                **annotate_props_defaults
                                ))

    elif plot in ['contour', 'contourf', 'histogram']:

        contour_props_defaults = {
                "cmap": mpl.cm.inferno,
            }

        if contour_props is not None:
            contour_props_defaults.update(contour_props)

        hist_props_defaults = {
            "avoid_zero_count": False,
            "mass": True,
            "mids": True
        }

        if hist_props is not None:
            hist_props_defaults.update(hist_props)

        avoid_zero_count = hist_props_defaults['avoid_zero_count']
        del hist_props_defaults['avoid_zero_count']

        mass = hist_props_defaults['mass']
        del hist_props_defaults['mass']

        mids = hist_props_defaults['mids']
        del hist_props_defaults['mids']

        if original:
            x_, y_, H = get_histogram(
                _data[:, 0], _data[:, 1],
                mids=mids,
                mass=mass,
                avoid_zero_count=avoid_zero_count,
                hist_props=hist_props_defaults
            )

            if free_energy:
                dG = np.inf * np.ones(shape=H.shape)

                nonzero = H.nonzero()
                dG[nonzero] = -np.log(H[nonzero])
                dG[nonzero] -= np.min(dG[nonzero])
                H = dG

            if plot == "histogram":
                # Plotting the histogram directly
                # imshow, pcolormesh, NonUniformImage ...
                # Not implemented, instead return the histogram
                warnings.warn(
                    "Plotting a histogram of the data directly is "
                    "currently not supported. Returning the edges and the "
                    "histogram instead.",
                    UserWarning
                )
                return x_, y_, H

            elif plot == 'contour':
                X, Y = np.meshgrid(x_, y_)
                plotted.append(
                    ax.contour(X, Y, H, **contour_props_defaults)
                    )

            elif plot == 'contourf':
                X, Y = np.meshgrid(x_, y_)
                plotted.append(
                    ax.contourf(X, Y, H, **contour_props_defaults)
                    )

            else:
                raise ValueError(
                    f"Plot type {plot} not understood. "
                    "Must be one of 'dots, 'scatter' or 'contour(f)'"
                    )

        else:
            for cluster, cpoints in items:
                if cluster in clusters:
                    x_, y_, H = get_histogram(
                        _data[cpoints, 0], _data[cpoints, 1],
                        mids=mids,
                        mass=mass,
                        avoid_zero_count=avoid_zero_count,
                        hist_props=hist_props_defaults
                    )

                    if free_energy:
                        dG = np.inf * np.ones(shape=H.shape)

                        nonzero = H.nonzero()
                        dG[nonzero] = -np.log(H[nonzero])
                        dG[nonzero] -= np.min(dG[nonzero])
                        H = dG

                    if plot == "histogram":
                        # Plotting the histogram directly
                        # imshow, pcolormesh, NonUniformImage ...
                        # Not implemented, instead return the histogram
                        warnings.warn(
    """Plotting a histogram of the data directly is currently not supported.
    Returning the edges and the histogram instead.
    """,
    UserWarning
                        )
                        return x_, y_, H

                    elif plot == 'contour':
                        X, Y = np.meshgrid(x_, y_)
                        plotted.append(
                            ax.contour(X, Y, H, **contour_props_defaults)
                            )

                    elif plot == 'contourf':
                        X, Y = np.meshgrid(x_, y_)
                        plotted.append(
                            ax.contourf(X, Y, H, **contour_props_defaults)
                            )

                    else:
                        raise ValueError(
            f"""Plot type {plot} not understood.
            Must be one of 'dots, 'scatter' or 'contour(f)'
            """
                )

    ax.set(**ax_props_defaults)

    return fig, ax, plotted


def get_histogram(
        x: Sequence[float], y: Sequence[float],
        mids: bool = True, mass: bool = True,
        avoid_zero_count: bool = True,
        hist_props: Optional[Dict['str', Any]] = None
        ) -> Tuple[np.ndarray, ...]:
    """Compute a two-dimensional histogram.

    Taken and modified from :module:`pyemma.plots.`

    Args:
        x: Sample x-coordinates.
        y: Sample y-coordinates.

    Keyword args:
        hist_props: Kwargs passed to `numpy.histogram2d`
        avoid_zero_count: Avoid zero counts by lifting all histogram
            elements to the minimum value before computing the free
            energy.  If False, zero histogram counts yield infinity in
            the free energy.
        mass: Norm the histogram by the total number of counts, so that
            each bin holds the probability mass values where all
            probabilities sum to 1
        mids: Return the mids of the bin edges instead of the actual
            edges

    Returns:
        The x- and y-edges and the data of the computed histogram

    """

    hist_props_defaults = {
        'bins': 100,
    }

    if hist_props is not None:
        hist_props_defaults.update(hist_props)

    z, x_, y_ = np.histogram2d(
        x, y, **hist_props_defaults
        )

    if mids:
        x_ = 0.5 * (x_[:-1] + x_[1:])
        y_ = 0.5 * (y_[:-1] + y_[1:])

    if avoid_zero_count:
        z = np.maximum(z, np.min(z[z.nonzero()]))

    if mass:
        z /= float(z.sum())

    return x_, y_, z.T  # transpose to match x/y-directions