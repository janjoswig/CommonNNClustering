"""Extension module containing utilities for plotting"""

import random
from typing import Dict, Tuple  # List, Set
from typing import Sequence  # Iterable, Iterator, Collection
from typing import Any, Optional  # Type, Union, IO

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def getpieces(c, pieces=None, level=0, ref="0", total=None):
    """Return cluster hierarchy structure in dict view

    Used e.g. by :meth:`plot_pie`.

    Pieces is a dictionary of the form
    dict[levels][clusterref] = {child: share}, e.g. :

        pieces = {
            0: {  # Level
                "0": {  # Reference
                    0: {0.1, 1: 0.9}
                },
            1: {
                "0.0": {
                    0: 0.1
                    },
                "0.1": {
                    0: 0.5,
                    1: 0.2,
                    2: 0.2
                    }
                }
            }

    Args:
        c: cluster object
        pieces: current view
        level: current hierarchy level
        ref: child cluster reference string
        total: number of point in root

    Returns:
        dict
    """

    if not pieces:
        # Init
        pieces = {}
    if level not in pieces:
        # New level
        pieces[level] = {}

    if c.labels.size > 0:
        # Build parts for current level
        cluster_shares = {k: len(v) for k, v in c.labels.clusterdict.items()}
        if total is None:
            # Only for root
            total = sum(cluster_shares.values())
        cluster_shares = {k: v / total for k, v in cluster_shares.items()}

        pieces[level][ref] = cluster_shares
        if level > 0:
            # Pad not reclustered clusters (as noise)
            for higher_ref, higher_cluster_shares in pieces[level - 1].items():
                for cluster, share in higher_cluster_shares.items():
                    entailed_ref = ".".join([higher_ref, str(cluster)])
                    if entailed_ref not in pieces[level]:
                        pieces[level][entailed_ref] = {0: share}

    if c._children:
        for number, child in c._children.items():
            pieces = getpieces(
                child,
                pieces=pieces,
                level=level + 1,
                ref=".".join([ref, str(number)]),
                total=total
            )

    return pieces


def pie(root, ax, pie_props=None):
    """Illustrate (hierarchichal) cluster result as pie diagram

    Args:
        root: :obj:`cnnclustering.cnn.CNN` being the origin of the pie diagram.
        ax: Matplotlib `Axes` instance to plot on.
        pie_props: Dictionary passed to :func:`matplotlib.pyplot.pie`.

    Returns:
        List of plotted elements (pie rings)
    """

    # TODO Make noise color configurable
    # TODO Adapt the scheme for tree view

    if ax is None:
        ax = plt.gca()

    pie_props_defaults = {
        "normalize": False,
        "radius": 0.5,
        "wedgeprops": dict(width=0.5, edgecolor='w')
        }

    if pie_props is not None:
        pie_props_defaults.update(pie_props)

    radius = pie_props_defaults.pop("radius")
    try:
        _ = pie_props_defaults.pop("colors")
    except KeyError:
        pass

    p = getpieces(root)

    plotted = []
    for level, refs in sorted(p.items()):
        ax.set_prop_cycle(None)
        ringvalues = []
        colors = []
        for ref, cluster_shares in sorted(refs.items()):
            for cluster, share in sorted(cluster_shares.items()):
                ringvalues.append(share)
                if cluster == 0:
                    colors.append("#262626")
                else:
                    colors.append(next(ax._get_lines.prop_cycler)["color"])

        plotted.append(
            ax.pie(ringvalues, radius=radius * (level + 1),
                   colors=colors, **pie_props_defaults),
            )

    return plotted


def plot_summary(
        ax, summary, quant="time", treat_nan=None,
        contour_props=None):
    """Generate a 2D plot of record values"""

    if contour_props is None:
        contour_props = {}

    pivot = summary.groupby(
        ["r", "c"]
        ).mean()[quant].reset_index().pivot(
            "r", "c"
            )

    X_, Y_ = np.meshgrid(
        pivot.index.values, pivot.columns.levels[1].values
        )

    values_ = pivot.values.T

    if treat_nan is not None:
        values_[np.isnan(values_)] == treat_nan

    plotted = []

    plotted.append(
        ax.contourf(X_, Y_, values_, **contour_props)
        )

    return plotted


def plot_dots(
        ax, data, original=True, clusterdict=None, clusters=None,
        dot_props=None, dot_noise_props=None,
        annotate=False, annotate_pos="mean", annotate_props=None):

    if dot_props is None:
        dot_props = {}

    if dot_noise_props is None:
        dot_noise_props = {}

    plotted = []

    if original:
        # Plot the original data
        plotted.append(
            ax.plot(
                data[:, 0],
                data[:, 1],
                **dot_props
                )
            )

    else:
        # Loop through the cluster result
        for cluster, cpoints in sorted(clusterdict.items()):
            # plot if cluster is in the list of considered clusters
            if cluster in clusters:
                cpoints = list(cpoints)

                # treat noise differently
                if cluster == 0:
                    plotted.append(ax.plot(
                        data[cpoints, 0],
                        data[cpoints, 1],
                        **dot_noise_props
                        ))

                else:
                    plotted.append(ax.plot(
                        data[cpoints, 0],
                        data[cpoints, 1],
                        **dot_props
                        ))

                    if annotate:
                        plotted.append(
                            annotate_points(
                                ax, annotate_pos, data, cpoints, cluster,
                                annotate_props
                                )
                            )
    return plotted


def plot_scatter(
        ax, data, original=True, clusterdict=None, clusters=None,
        scatter_props=None, scatter_noise_props=None,
        annotate=False, annotate_pos="mean", annotate_props=None):

    if scatter_props is None:
        scatter_props = {}

    if scatter_noise_props is None:
        scatter_noise_props = {}

    plotted = []

    if original:
        plotted.append(
            ax.scatter(
                data[:, 0],
                data[:, 1],
                **scatter_props
                )
            )

    else:
        for cluster, cpoints in sorted(clusterdict.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                # treat noise differently
                if cluster == 0:
                    plotted.append(
                        ax.scatter(
                            data[cpoints, 0],
                            data[cpoints, 1],
                            **scatter_noise_props
                            )
                        )

                else:
                    plotted.append(
                        ax.scatter(
                            data[cpoints, 0],
                            data[cpoints, 1],
                            **scatter_props
                            )
                        )

                    if annotate:
                        plotted.append(
                            annotate_points(
                                ax, annotate_pos, data, cpoints, cluster,
                                annotate_props
                                )
                            )
    return plotted


def plot_contour(
        ax, data, original=True, clusterdict=None, clusters=None,
        contour_props=None, contour_noise_props=None,
        hist_props=None, free_energy=True,
        annotate=False, annotate_pos="mean", annotate_props=None):

    if contour_props is None:
        contour_props = {}

    if contour_noise_props is None:
        contour_noise_props = {}

    if hist_props is None:
        hist_props = {}

    if 'avoid_zero_count' in hist_props:
        avoid_zero_count = hist_props['avoid_zero_count']
        del hist_props['avoid_zero_count']

    if 'mass' in hist_props:
        mass = hist_props['mass']
        del hist_props['mass']

    if 'mids' in hist_props:
        mids = hist_props['mids']
        del hist_props['mids']

    plotted = []

    if original:
        x_, y_, H = get_histogram(
            data[:, 0], data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props
        )

        if free_energy:
            H = get_free_energy(H)

        X, Y = np.meshgrid(x_, y_)
        plotted.append(
            ax.contour(X, Y, H, **contour_props)
            )
    else:
        for cluster, cpoints in sorted(clusterdict.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram(
                    data[cpoints, 0], data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contour(X, Y, H, **contour_noise_props)
                        )
                else:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contour(X, Y, H, **contour_props)
                        )

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                            )
                        )

    return plotted


def plot_contourf(
        ax, data, original=True, clusterdict=None, clusters=None,
        contour_props=None, contour_noise_props=None,
        hist_props=None, free_energy=True,
        annotate=False, annotate_pos="mean", annotate_props=None):

    if contour_props is None:
        contour_props = {}

    if contour_noise_props is None:
        contour_noise_props = {}

    if hist_props is None:
        hist_props = {}

    if 'avoid_zero_count' in hist_props:
        avoid_zero_count = hist_props['avoid_zero_count']
        del hist_props['avoid_zero_count']

    if 'mass' in hist_props:
        mass = hist_props['mass']
        del hist_props['mass']

    if 'mids' in hist_props:
        mids = hist_props['mids']
        del hist_props['mids']

    plotted = []

    if original:
        x_, y_, H = get_histogram(
            data[:, 0], data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props
        )

        if free_energy:
            H = get_free_energy(H)

        X, Y = np.meshgrid(x_, y_)
        plotted.append(
            ax.contourf(X, Y, H, **contour_props)
            )
    else:
        for cluster, cpoints in sorted(clusterdict.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram(
                    data[cpoints, 0], data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contourf(X, Y, H, **contour_noise_props)
                        )
                else:
                    X, Y = np.meshgrid(x_, y_)
                    plotted.append(
                        ax.contourf(X, Y, H, **contour_props)
                        )

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                            )
                        )

    return plotted


def plot_histogram(
        ax, data, original=True, clusterdict=None, clusters=None,
        show_props=None, show_noise_props=None,
        hist_props=None, free_energy=True,
        annotate=False, annotate_pos="mean", annotate_props=None):

    if show_props is None:
        show_props = {}

    if show_noise_props is None:
        show_noise_props = {}

    if 'extent' in show_props:
        del show_props['extent']

    if hist_props is None:
        hist_props = {}

    if 'avoid_zero_count' in hist_props:
        avoid_zero_count = hist_props['avoid_zero_count']
        del hist_props['avoid_zero_count']

    if 'mass' in hist_props:
        mass = hist_props['mass']
        del hist_props['mass']

    if 'mids' in hist_props:
        mids = hist_props['mids']
        del hist_props['mids']

    plotted = []

    if original:
        x_, y_, H = get_histogram(
            data[:, 0], data[:, 1],
            mids=mids,
            mass=mass,
            avoid_zero_count=avoid_zero_count,
            hist_props=hist_props
        )

        if free_energy:
            H = get_free_energy(H)

        plotted.append(
            ax.imshow(H, extent=(x_, y_), **show_props)
            )
    else:
        for cluster, cpoints in sorted(clusterdict.items()):
            if cluster in clusters:
                cpoints = list(cpoints)

                x_, y_, H = get_histogram(
                    data[cpoints, 0], data[cpoints, 1],
                    mids=mids,
                    mass=mass,
                    avoid_zero_count=avoid_zero_count,
                    hist_props=hist_props
                )

                if free_energy:
                    H = get_free_energy(H)

                if cluster == 0:
                    plotted.append(
                        ax.imshow(H, extent=(x_, y_), **show_noise_props)
                        )
                else:
                    plotted.append(
                        ax.imshow(H, extent=(x_, y_), **show_props)
                        )

                if annotate:
                    plotted.append(
                        annotate_points(
                            ax, annotate_pos, data, cpoints, cluster,
                            annotate_props
                            )
                        )
    return plotted


def annotate_points(ax, pos, data, points, text, annotate_props=None):
    if annotate_props is None:
        annotate_props = {}

    if pos == "mean":
        xpos = np.mean(data[points, 0])
        ypos = np.mean(data[points, 1])

    elif pos == "random":
        choosen = random.sample(
            points, 1
            )
        xpos = data[choosen, 0]
        ypos = data[choosen, 1]

    else:
        raise ValueError(
            'Keyword argument `annotate_pos` must be '
            'one of "mean", "random"'
            )

    return ax.annotate(
        f"{text}",
        xy=(xpos, ypos),
        **annotate_props
        )


def get_free_energy(H):
    dG = np.inf * np.ones(shape=H.shape)

    nonzero = H.nonzero()
    dG[nonzero] = -np.log(H[nonzero])
    dG[nonzero] -= np.min(dG[nonzero])

    return dG


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
