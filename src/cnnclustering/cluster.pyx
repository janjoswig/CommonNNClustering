from collections import Counter
from collections.abc import MutableMapping
import functools
from operator import itemgetter
import time
from typing import Any, Optional, Type, Union
from typing import Container, Iterable, List, Tuple, Sequence
import weakref

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from . import plot
    MPL_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    MPL_FOUND = False

try:
    import networkx as nx
    NX_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    NX_FOUND = False

import numpy as np
cimport numpy as np

try:
    import pandas as pd
    PANDAS_FOUND = True
except ModuleNotFoundError as error:
    print("Optional dependency module not found: ", error)
    PANDAS_FOUND = False

from cnnclustering._primitive_types import P_AINDEX, P_AVALUE, P_ABOOL
from cnnclustering._fit import Fitter, HierarchicalFitter, Predictor
from cnnclustering._types import InputData
from cnnclustering import recipes
from cnnclustering.report import Record, Summary

from libcpp.vector cimport vector as cppvector
from libcpp.unordered_map cimport unordered_map as cppumap


class Clustering:
    r"""Represents a clustering endeavour

    A clustering object is made by aggregation of all necessary parts to
    carry out a clustering of input data points.

    Keyword args:
        data:
            The data points to be clustered. Can be one of

                * `None`:
                    Plain initialisation without input data.
                * A :class:`~cnnclustering._bundle.Bundle`:
                    Initialisation with a ready-made input data bundle.
                * Any object implementing the input data data interface
                  (see :class:`~cnnclustering._types.InputData` or
                  :class:`~cnnclustering._types.InputDataExtInterface`):
                    In this case, additional keyword arguments can be passed
                    via `bundle_kwargs` which are used to initialise a
                    :class:`~cnnclustering._bundle.Bundle` from the input data,
                    e.g. `labels`, `children`, etc.
                * Raw input data: Takes the input data type and a preparation
                  hook from the `recipe` and wraps the raw data.

        fitter:
            Executes the clustering procedure. Can be

                * Any object implementing the fitter interface (see :class:`~cnnclustering._fit.Fitter` or
                  :class:`~cnnclustering._fit.FitterExtInterface`).
                * None:
                    In this case, the fitter is tried to be build from the `recipe` or left
                    as `None`.

        hierarchical_fitter:
            Like `fitter` but for hierarchical clustering (see
            :class:`~cnnclustering._fit.HierarchicalFitter` or
            :class:`~cnnclustering._fit.HierarchicalFitterExtInterface`).

        predictor:
            Translates a clustering result from one bundle to another. Treated like
            `fitter` (see
            :class:`~cnnclustering._fit.Predictor` or
            :class:`~cnnclustering._fit.PredictorExtInterface`).

        bundle_kwargs: Used to create a :class:`~cnnclustering._bundle.Bundle`
            if `data` is neither a bundle nor `None`.

        recipe:
            Used to assemble a fitter etc. and to wrap raw input data. Can be

                * A string corresponding to a registered default recipe (see
                    :obj:`~cnnclustering.recipes.REGISTERED_RECIPES`
                )
                * A recipe, i.e. a mapping of component keywords to component types
    """

    def __init__(
            self,
            data=None, *,  # TODO use positional only modifier "/" (Python >= 3.8)
            fitter=None,
            hierarchical_fitter=None,
            predictor=None,
            bundle_kwargs=None,
            recipe=None,
            **recipe_kwargs):

        builder = recipes.Builder(recipe, **recipe_kwargs)

        if bundle_kwargs is None:
            bundle_kwargs = {}

        if data is None:
            self._bundle = None

        elif isinstance(data, Bundle):
            self._bundle = data

        elif isinstance(data, InputData):
            if bundle_kwargs is None:
                bundle_kwargs = {}

            self._bundle = Bundle(data, **bundle_kwargs)
        else:
            # TODO: Guess input data type and preparation hook
            data = builder.make_input_data(data)
            self._bundle = Bundle(data, **bundle_kwargs)

        for kw, component_kw, kw_type in [
                (fitter, "fitter", Fitter),
                (hierarchical_fitter, "hierarchical_fitter", HierarchicalFitter),
                (predictor, "predictor", Predictor)
                ]:

            if isinstance(kw, kw_type):
                setattr(self, f"_{component_kw}", kw)

            elif kw is None:
                kw = builder.make_component(component_kw)
                if kw is object: kw = None
                setattr(self, f"_{component_kw}", kw)

            else:
                raise TypeError(
                    f"Object {fitter} is not valid for 'fitter'"
                    )

    @property
    def root(self):
        if self._bundle is None:
            return
        return self._bundle

    @property
    def labels(self):
        """
        Direct access to :obj:`~cnnclustering._types.Labels.labels`
        holding cluster label assignments for points in :obj:`~cnnclustering._types.InputData`
        stored on the root :obj:`~cnnclustering._bundle.Bundle`.
        """
        if self._bundle is None:
            return None
        if self._bundle._labels is None:
            return None
        return self._bundle._labels.labels

    @labels.setter
    def labels(self, value):
        """
        Direct access to :obj:`~cnnclustering._types.Labels`
        holding cluster label assignments for points in :obj:`~cnnclustering._types.InputData`
        stored on the root :obj:`~cnnclustering._bundle.Bundle`.
        """
        if self._bundle is None:
            raise ValueError("Can't set labels because there is no root bundle")
        self._bundle.labels = value

    @property
    def input_data(self):
        if self._bundle is None:
            return None
        if self._bundle._input_data is None:
            return None
        return self._bundle._input_data.data

    @property
    def fitter(self):
        return self._fitter

    @fitter.setter
    def fitter(self, value):
        if (value is not None) & (not isinstance(value, Fitter)):
            raise TypeError(
                f"Can't use object of type {type(value).__name__} as fitter. "
                f"Expected type {Fitter.__name__}."
                )
        self._fitter = value

    @property
    def children(self):
        """
        Return a mapping of child cluster labels to
        :obj:`cnnclustering.cluster.Clustering` instances representing
        the children of this clustering.
        """
        return self._bundle._children

    @property
    def summary(self):
        """
        Return an instance of :obj:`cnnclustering.cluster.Summary`
        collecting clustering results for this clustering.
        """
        return self._bundle._summary

    def __str__(self):
        attr_str = ", ".join([
            f"input_data={self._bundle._input_data}",
            f"fitter={self._fitter}",
            f"hfitter={self._hierarchical_fitter}",
            f"predictor={self._predictor}",
        ])

        return f"{type(self).__name__}({attr_str})"

    def __getitem__(self, value):
        return self._bundle.get_child(value)

    def _fit(
            self,
            cluster_params: Type["ClusterParameters"],
            bundle=None) -> None:
        """Execute clustering procedure

        Low-level alternative to
        :meth:`cnnclustering.cluster.Clustering.fit`.

        Note:
            No pre-processing of cluster parameters (radius and
            similarity value adjustements based on used metric,
            neighbours getter, etc.) and label initialisation is done
            before the fit. It is the users responsibility to take care
            of this to obtain sensible results.
        """

        if bundle is None:
            bundle = self._bundle

        self._fitter.fit(
            bundle._input_data,
            bundle._labels,
            cluster_params
            )

    def fit(
            self,
            radius_cutoff: float,
            cnn_cutoff: int,
            member_cutoff: int = None,
            max_clusters: int = None,
            cnn_offset: int = None,
            sort_by_size: bool = True,
            info: bool = True,
            record: bool = True,
            record_time: bool = True,
            v: bool = True,
            purge: bool = False,
            bundle=None) -> None:
        """Execute clustering procedure

        Args:
            radius_cutoff: Neighbour search radius.
            cnn_cutoff: Similarity criterion.
            member_cutoff: Valid clusters need to have at least this
                many members.  Passed on  to :meth:`Labels.sort_by_size`
                if `sort_by_size` is `True`.  Has no effect otherwise
                and valid clusters have at least one member.
            max_clusters: Keep only the largest `max_clusters` clusters.
                Passed on to :meth:`Labels.sort_by_size` if
                `sort_by_size` is `True`.  Has no effect otherwise.
            cnn_offset: Exists for compatibility reasons and is
                substracted from `cnn_cutoff`.  If `cnn_offset = 0`, two
                points need to share at least `cnn_cutoff` neighbours
                to be part of the same cluster without counting any of
                the two points.  In former versions of the clustering,
                self-counting was included and `cnn_cutoff = 2` is
                equivalent to `cnn_cutoff = 0` in this version.
            sort_by_size: Weather to sort (and trim) the created
                :obj:`Labels` instance.  See also
                :meth:`Labels.sort_by_size`.
            info: Wether to modify :obj:`Labels.meta` information for
                this clustering.
            record: Wether to create a :obj:`Record`
                instance for this clustering which is appended to the
                :obj:`Summary`.
            record_time: Wether to time clustering execution.
            v: Be chatty.
            purge: If True, force re-initialisation of cluster label
                assignments.
        """

        cdef set old_label_set, new_label_set
        cdef ClusterParameters cluster_params
        cdef AINDEX current_start

        if bundle is None:
            bundle = self._bundle

        if cnn_offset is None:
            cnn_offset = 0
        cnn_cutoff -= cnn_offset

        assert cnn_cutoff >= 0

        if (bundle._labels is None) or purge or (
                not bundle._labels.meta.get("frozen", False)):

            bundle._labels = Labels(
                np.zeros(bundle._input_data.n_points, order="C", dtype=P_AINDEX)
                )
            old_label_set = set()
            current_start = 1
        else:
            old_label_set = bundle._labels.to_set()
            current_start = max(old_label_set) + 1

        cluster_params = self._fitter.make_parameters(
            radius_cutoff, cnn_cutoff, current_start
            )

        if record_time:
            _, execution_time = timed(self._fitter.fit)(
                bundle._input_data,
                bundle._labels,
                cluster_params
                )
        else:
            self._fitter.fit(
                bundle._input_data,
                bundle._labels,
                cluster_params
                )
            execution_time = None

        if info:
            new_label_set = bundle._labels.to_set()
            params = {
                k: (radius_cutoff, cnn_cutoff)
                for k in new_label_set - old_label_set
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(bundle),
                "origin": "fit"
            }
            old_params = bundle._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            bundle._labels.meta.update(meta)

        if sort_by_size:
            bundle._labels.sort_by_size(member_cutoff, max_clusters)

        if record:
            n_noise = 0
            frequencies = Counter(bundle._labels.labels)

            if 0 in frequencies:
                n_noise = frequencies.pop(0)

            n_largest = frequencies.most_common(1)[0][1] if frequencies else 0

            rec = Record(
                bundle._input_data.n_points,
                radius_cutoff,
                cnn_cutoff,
                member_cutoff,
                max_clusters,
                len(bundle._labels.to_set() - {0}),
                n_largest / bundle._input_data.n_points,
                n_noise / bundle._input_data.n_points,
                execution_time,
                )

            if v:
                print(rec)

            bundle._summary.append(rec)

        return

    def fit_hierarchical(
            self,
            purge=True,
            bundle=None,
            **fitter_kwargs):
        """Execute hierarchical clustering procedure

        Keyword args:
            depend on hfitter
        """

        if bundle is None:
            bundle = self._bundle

        if purge or (bundle._children is None):
            bundle._children = {}

        self._hfitter.fit(bundle, **fitter_kwargs)

    def isolate(
            self,
            bint purge: bool = True,
            bint isolate_input_data: bool = True,
            bundle=None):
        """Create child clusterings from cluster labels

        Args:
            purge: If `True`, creates a new mapping for the children of this
                clustering.
            isolate_input_data: If `True`, attaches a subset of the input data
                of this clustering to the child.
            bundle: A bundle to operate on. If `None` uses the root bundle.
        """

        if bundle is None:
            bundle = self._bundle

        bundle.isolate(purge, isolate_input_data)

        return

    def trim_shrinking_leafs(self, bundle=None):

        if bundle is None:
            bundle = self._bundle

        def _trim_shrinking(bundle, new=True):

            if not bundle._children:
                splits = will_split = False
            else:
                label_set = bundle._labels.set
                label_set.discard(0)

                if len(label_set) <= 1:
                    splits = False
                else:
                    splits = True

                will_split = []
                for child in bundle._children.values():
                    will_split.append(
                        _trim_shrinking(child, new=splits)
                        )

                will_split = any(will_split)

            keep = new or will_split or splits
            if not keep:
                bundle._labels = None
                bundle._children = {}

            return keep

        _trim_shrinking(bundle)

        return

    def trim_trivial_leafs(self, bundle=None):
        """Scan cluster hierarchy for removable nodes

        If the cluster label assignments on a clustering are all zero
        (noise), the clustering is considered trivial.  In this case,
        the labels and children are reset to `None`.
        """

        if bundle is None:
            bundle = self._bundle

        def _trim_trivial(clustering):
            if clustering._labels is None:
                return

            if clustering._labels.set == {0}:
                clustering._labels = None
                clustering._children = {}
                return

            if not clustering._children:
                return

            for child in clustering._children.values():
                _trim_trivial(child)

        _trim_trivial(bundle)

        return

    def reel(self, depth: Optional[int] = None, bundle=None) -> None:
        """Wrap up label assignments of lower hierarchy levels

        Args:
            depth: How many lower levels to consider. If `None`,
            consider all.
        """

        if bundle is None:
            bundle = self._bundle

        cdef AINDEX label, new_label, parent_index

        def _reel(parent, depth):
            if not parent._children:
                return

            if depth is not None:
                depth -= 1

            parent._labels.meta["origin"] = "reel"
            parent_labels = parent._labels.labels

            for label, child in parent._children.items():
                if child._labels is None:
                    continue

                if (depth is None) or (depth > 0):
                    _reel(child, depth)

                n_clusters = max(parent_labels)

                child_labels = child._labels.labels
                for index, old_label in enumerate(child_labels):
                    if old_label == 0:
                        new_label = 0
                    else:
                        new_label = old_label + n_clusters

                    parent_index = child.parent_indices[index]
                    parent_labels[parent_index] = new_label

                try:
                    _ = parent._labels.meta["params"].pop(label)
                except KeyError:
                    pass

                params = child._labels.meta.get("params", {})
                for old_label, p in params.items():
                    parent._labels.meta["params"][old_label + n_clusters] = p

        if depth is not None:
            assert depth > 0

        _reel(bundle, depth)

        return

    def predict(
            self,
            other,
            radius_cutoff: float,
            cnn_cutoff: int,
            clusters: Optional[Sequence[int]] = None,
            cnn_offset: Optional[int] = None,
            info: bool = True,
            record: bool = True,
            record_time: bool = True,
            v: bool = True,
            purge: bool = False,
            bundle=None):
        """Execute prediction procedure

        Args:
            other: :obj:`cnnclustering.cluster.Clustering` instance for
                which cluster labels should be predicted.
            radius_cutoff: Neighbour search radius.
            cnn_cutoff: Similarity criterion.
            cluster: Sequence of cluster labels that should be included
               in the prediction.
            cnn_offset: Exists for compatibility reasons and is
                substracted from `cnn_cutoff`.  If `cnn_offset = 0`, two
                points need to share at least `cnn_cutoff` neighbours
                to be part of the same cluster without counting any of
                the two points.  In former versions of the clustering,
                self-counting was included and `cnn_cutoff = 2` is
                equivalent to `cnn_cutoff = 0` in this version.
            purge: If True, force re-initialisation of predicted cluster
                labels.
        """

        if bundle is None:
            bundle = self._bundle

        cdef ClusterParameters cluster_params

        if cnn_offset is None:
            cnn_offset = 0
        cnn_cutoff -= cnn_offset

        assert cnn_cutoff >= 0

        if (other._labels is None) or purge or (
                not other._labels.meta.get("frozen", False)):
            other._labels = Labels(
                np.zeros(other._input_data.n_points, order="C", dtype=P_AINDEX)
                )

        cluster_params = self._predictor.make_parameters(
            radius_cutoff, cnn_cutoff, 0
            )

        if clusters is None:
           clusters = bundle._labels.to_set() - {0}

        other._labels.consider_set = clusters

        if record_time:
            _, execution_time = timed(self._predictor.predict)(
                bundle._input_data,
                other._input_data,
                bundle._labels,
                other._labels,
                cluster_params
                )
        else:
            self._predictor.predict(
                bundle._input_data,
                other._input_data,
                bundle._labels,
                other._labels,
                cluster_params
                )
            execution_time = None

        if info:
            params = {
                k: (radius_cutoff, cnn_cutoff)
                for k in clusters
                if k != 0
                }
            meta = {
                "params": params,
                "reference": weakref.proxy(bundle),
                "origin": "predict",
            }
            old_params = other._labels.meta.get("params", {})
            old_params.update(meta["params"])
            meta["params"] = old_params
            other._labels.meta.update(meta)

        other._labels.meta["frozen"] = True

    def summarize(
            self,
            ax=None,
            quantity: str = "execution_time",
            treat_nan: Optional[Any] = None,
            convert: Optional[Any] = None,
            ax_props: Optional[dict] = None,
            contour_props: Optional[dict] = None,
            plot_style: str = "contourf",
            bundle=None):
        """Generate a 2D plot of record values

        Record values ("time", "clusters", "largest", "noise") are
        plotted against cluster parameters (radius cutoff *r*
        and cnn cutoff *c*).

        Args:
            ax: Matplotlib Axes to plot on.  If `None`, a new Figure
                with Axes will be created.
            quantity: Record value to
                visualise:

                    * "time"
                    * "clusters"
                    * "largest"
                    * "noise"

            treat_nan: If not `None`, use this value to pad nan-values.
            ax_props: Used to style `ax`.
            contour_props: Passed on to contour.
        """

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        if (self._bundle._summary is None) or (len(self._bundle._summary._list) == 0):
            raise LookupError(
                "No records in summary"
                )

        ax_props_defaults = {
            "xlabel": "$r$",
            "ylabel": "$c$",
        }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        contour_props_defaults = {
                "cmap": mpl.cm.inferno,
            }

        if contour_props is not None:
            contour_props_defaults.update(contour_props)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plotted = plot.plot_summary(
            ax, bundle._summary.to_DataFrame(),
            quantity=quantity,
            treat_nan=treat_nan,
            convert=convert,
            contour_props=contour_props_defaults,
            plot_style=plot_style,
            )

        ax.set(**ax_props_defaults)

        return plotted

    def pie(self, ax=None, pie_props=None, bundle=None):

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        plot.pie(bundle, ax=ax, pie_props=pie_props)

        return

    def tree(
            self,
            ax=None,
            ignore=None,
            pos_props=None,
            draw_props=None,
            bundle=None):

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if bundle is None:
            bundle = self._bundle

        graph = self.to_nx_DiGraph(ignore=ignore, bundle=bundle)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        pos_props_defaults = {
            "source": "1",
        }

        if pos_props is not None:
            pos_props_defaults.update(pos_props)

        shortened_labels = {}
        for key in graph.nodes.keys():
            skey = key.rsplit(".", 1)
            shortened_labels[key] = skey[len(skey) - 1]

        draw_props_defaults = {
            "labels": shortened_labels,
            "with_labels": True,
            "node_shape": "s",
            "edgecolors": "k",
        }

        if draw_props is not None:
            draw_props_defaults.update(draw_props)

        plot.plot_graph_sugiyama_straight(
            graph, ax=ax,
            pos_props=pos_props_defaults,
            draw_props=draw_props_defaults,
            )

        return

    def evaluate(
            self,
            ax=None,
            clusters: Optional[Container[int]] = None,
            original: bool = False,
            plot_style: str = 'dots',
            parts: Optional[Tuple[Optional[int]]] = None,
            points: Optional[Tuple[Optional[int]]] = None,
            dim: Optional[Tuple[int, int]] = None,
            mask: Optional[Sequence[Union[bool, int]]] = None,
            ax_props: Optional[dict] = None,
            annotate: bool = True,
            annotate_pos: Union[str, dict] = "mean",
            annotate_props: Optional[dict] = None,
            plot_props: Optional[dict] = None,
            plot_noise_props: Optional[dict] = None,
            hist_props: Optional[dict] = None,
            free_energy: bool = True,
            bundle=None):

        """Returns a 2D plot of an original data set or a cluster result

        Args: ax: The `Axes` instance to which to add the plot.  If
            `None`, a new `Figure` with `Axes` will be created.

            clusters:
                Cluster numbers to include in the plot.  If `None`,
                consider all.

            original:
                Allows to plot the original data instead of a cluster
                result.  Overrides `clusters`.  Will be considered
                `True`, if no cluster result is present.

            plot_style:
                The kind of plotting method to use:

                    * "dots", :func:`ax.plot`
                    * "scatter", :func:`ax.scatter`
                    * "contour", :func:`ax.contour`
                    * "contourf", :func:`ax.contourf`

            parts:
                Use a slice (start, stop, stride) on the data parts
                before plotting. Will be applied before a slice on `points`.

            points:
                Use a slice (start, stop, stride) on the data points
                before plotting.

            dim:
                Use these two dimensions for plotting.  If `None`, uses
                (0, 1).

            mask:
                Sequence of boolean or integer values used for optional
                fancy indexing on the point data array.  Note, that this
                is applied after regular slicing (e.g. via `points`) and
                requires a copy of the indexed data (may be slow and
                memory intensive for big data sets).

            annotate:
                If there is a cluster result, plot the cluster numbers.
                Uses `annotate_pos` to determinte the position of the
                annotations.

            annotate_pos:
                Where to put the cluster number annotation.
                Can be one of:

                    * "mean", Use the cluster mean
                    * "random", Use a random point of the cluster
                    * dict `{1: (x, y), ...}`, Use a specific coordinate
                        tuple for each cluster. Omitted labels will be placed
                        randomly.

            annotate_props:
                Dictionary of keyword arguments passed to
                :func:`ax.annotate`.

            ax_props:
                Dictionary of `ax` properties to apply after
                plotting via :func:`ax.set(**ax_props)`.  If `None`,
                uses defaults that can be also defined in
                the configuration file (*Note yet implemented*).

            plot_props:
                Dictionary of keyword arguments passed to various
                functions (:func:`plot.plot_dots` etc.) with different
                meaning to format cluster plotting.  If `None`, uses
                defaults that can be also defined in
                the configuration file (*Note yet implemented*).

            plot_noise_props:
                Like `plot_props` but for formatting noise point
                plotting.

            hist_props:
               Dictionary of keyword arguments passed to functions that
               involve the computing of a histogram via
               `numpy.histogram2d`.

            free_energy:
                If `True`, converts computed histograms to pseudo free
                energy surfaces.

        Returns:
            Figure, Axes and a list of plotted elements
        """

        if bundle is None:
            bundle = self._bundle

        if not MPL_FOUND:
            raise ModuleNotFoundError("No module named 'matplotlib'")

        if (bundle._input_data is None) or (
                not bundle._input_data.meta.get("access_coords", False)):
            raise ValueError(
                "No data point coordinates found to evaluate."
            )

        if dim is None:
            dim = (0, 1)
        elif dim[1] < dim[0]:
            dim = dim[::-1]  # Problem with wraparound=False?

        if parts is not None:
            by_parts = list(bundle._input_data.by_parts())[slice(*parts)]
            data = np.vstack(by_parts)
        else:
            data = bundle._input_data.data

        if points is None:
            points = (None, None, None)

        # Slicing without copying
        data = data[
            slice(*points),
            slice(dim[0], dim[1] + 1, dim[1] - dim[0])
            ]

        if mask is not None:
            data = data[mask]

        # Plot original set or points per cluster?
        cluster_map = None
        if not original:
            if bundle._labels is not None:
                cluster_map = bundle._labels.mapping
                if clusters is None:
                    clusters = list(cluster_map.keys())
            else:
                original = True

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

        if plot_style == "dots":
            plot_props_defaults = {
                'lw': 0,
                'marker': '.',
                'markersize': 5,
                'markeredgecolor': 'none',
                }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'none',
                'lw': 0,
                'marker': '.',
                'markersize': 4,
                'markerfacecolor': 'k',
                'markeredgecolor': 'none',
                'alpha': 0.3
                }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_dots(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                dot_props=plot_props_defaults,
                dot_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        elif plot_style == "scatter":
            plot_props_defaults = {
                's': 10,
            }

            if plot_props is not None:
                plot_props_defaults.update(plot_props)

            plot_noise_props_defaults = {
                'color': 'k',
                's': 10,
                'alpha': 0.5
            }

            if plot_noise_props is not None:
                plot_noise_props_defaults.update(plot_noise_props)

            plotted = plot.plot_scatter(
                ax=ax, data=data, original=original,
                cluster_map=cluster_map,
                clusters=clusters,
                scatter_props=plot_props_defaults,
                scatter_noise_props=plot_noise_props_defaults,
                annotate=annotate, annotate_pos=annotate_pos,
                annotate_props=annotate_props_defaults
                )

        if plot_style in ["contour", "contourf", "histogram"]:

            hist_props_defaults = {
                "avoid_zero_count": False,
                "mass": True,
                "mids": True
            }

            if hist_props is not None:
                hist_props_defaults.update(hist_props)

            if plot_style == "contour":

                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contour(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "contourf":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_contourf(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

            elif plot_style == "histogram":
                plot_props_defaults = {
                    "cmap": mpl.cm.inferno,
                }

                if plot_props is not None:
                    plot_props_defaults.update(plot_props)

                plot_noise_props_defaults = {
                    "cmap": mpl.cm.Greys,
                }

                if plot_noise_props is not None:
                    plot_noise_props_defaults.update(plot_noise_props)

                plotted = plot.plot_histogram2d(
                    ax=ax, data=data, original=original,
                    cluster_map=cluster_map,
                    clusters=clusters,
                    contour_props=plot_props_defaults,
                    contour_noise_props=plot_noise_props_defaults,
                    hist_props=hist_props_defaults, free_energy=free_energy,
                    annotate=annotate, annotate_pos=annotate_pos,
                    annotate_props=annotate_props_defaults
                    )

        ax.set(**ax_props_defaults)

        return plotted

    def to_nx_DiGraph(self, ignore=None, bundle=None):
        """Convert cluster hierarchy to networkx DiGraph

        Keyword args:
            ignore: A set of label not to include into the graph.  Use
                for example to exclude noise (label 0).
        """

        if not NX_FOUND:
            raise ModuleNotFoundError("No module named 'networkx'")

        if bundle is None:
            bundle = self._bundle

        def add_children(clustering_label, clustering, graph):
            for child_label, child_clustering in sorted(clustering._children.items()):

                if child_label in ignore:
                    continue

                padded_child_label = ".".join([clustering_label, str(child_label)])
                graph.add_node(padded_child_label, object=child_clustering)
                graph.add_edge(clustering_label, padded_child_label)

                if child_clustering._children:
                    add_children(padded_child_label, child_clustering, graph)

        if ignore is None:
            ignore = set()

        if not isinstance(ignore, set):
            ignore = set(ignore)

        graph = nx.DiGraph()
        graph.add_node("1", object=bundle)
        add_children("1", bundle, graph)

        return graph

    def to_dtrajs(self, bundle=None):

        if bundle is None:
            bundle = self._bundle

        labels_array = bundle.labels
        if labels_array is None:
            return []

        edges = None
        if bundle._input_data is not None:
            edges = self._input_data.meta.get("edges")

        if edges is None:
            return [labels_array]

        dtrajs = np.split(labels_array, np.cumsum(edges))

        last_dtraj_index = len(dtrajs) - 1
        if len(dtrajs[last_dtraj_index]) == 0:
            dtrajs = dtrajs[:last_dtraj_index]

        return dtrajs


def timed(function):
    """Decorator to measure execution time"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        go = time.time()
        wrapped_return = function(*args, **kwargs)
        stop = time.time()

        return wrapped_return, stop - go
    return wrapper
