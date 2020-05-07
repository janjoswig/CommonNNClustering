# CLuster cleanup

clusters_no_noise = {
    y: self._clusterdict[y]
    for y in self._clusterdict if y != 0
    }

too_small = [
    self._clusterdict.pop(y)
    for y in [x[0]
    for x in clusters_no_noise.items() if len(x[1]) <= member_cutoff]
    ]

if len(too_small) > 0:
    for entry in too_small:
        self._clusterdict[0].update(entry)

for x in set(self._labels):
    if x not in set(self._clusterdict):
        self._labels[self._labels == x] = 0

if len(clusters_no_noise) == 0:
    largest = 0
else:
    largest = len(self._clusterdict[1 + np.argmax([
        len(x)
        for x in clusters_no_noise.values()
            ])]) / self.data.shape_str["points"][0]

# print(f"Found largest cluster: {time.time() - go}")

self._clusterdict = self._clusterdict
self._labels = self._labels
self.clean()
self.labels2dict()


# Record
cresult = TypedDataFrame(
    self.record._fields,
    self._record_dtypes,
    content=[
        [self.data.shape_str["points"][0]],
        [params["radius_cutoff"]],
        [params["cnn_cutoff"]],
        [params["member_cutoff"]],
        [params["max_clusters"]],
        [max(self._labels)],
        [len(np.where(self._labels == max(self._labels))[1]) / self.data.shape_str["points"][0]],
        [len(np.where(self._labels == 0)[0]) / self.data.shape_str["points"][0]],
        [None],
        ],
    )

if v:
    print("\n" + "-"*72)
    print(
        cresult[list(self.record._fields)[:-1]].to_string(
            na_rep="None", index=False, line_width=80,
            header=[
                "  #points  ", "  R  ", "  N  ", "  M  ",
                "  max  ", "  #clusters  ", "  %largest  ",
                "  %noise  "
                ],
            justify="center"
            ))
    print("-"*72)

    def isolate(self, purge=True):
        """Isolates points per clusters based on a cluster result"""

        if purge or self._children is None:
            self._children = defaultdict(lambda: CNNChild(self))

        for label, cpoints in self.labels.clusterdict.items():

            cpoints = list(cpoints)
            ref_index = []
            ref_index_rel = []
            cluster_data = []
            part_startpoint = 0

            if self._refindex is None:
                ref_index.extend(cpoints)
                ref_index_rel = ref_index
            else:
                ref_index.extend(self._refindex[cpoints])
                ref_index_rel.extend(cpoints)

            for part in range(self._shape['parts']):
                part_endpoint = part_startpoint \
                    + self._shape['points'][part] - 1

                cluster_data.append(
                    self._data[part][cpoints[
                        np.where(
                            (cpoints
                             >= part_startpoint)
                            &
                            (cpoints
                             <= part_endpoint))[0]] - part_startpoint]
                        )
                part_startpoint = np.copy(part_endpoint)
                part_startpoint += 1

            self._children[label].alias = f'child No. {label}'
            self._children[label].data.points = cluster_data
            self._children[label]._refindex = np.asarray(ref_index)
            self._children[label]._refindex_rel = np.asarray(ref_index_rel)
        return

    def loop_over_points(self) -> Iterator:
        """Iterate over all points of all parts

        Returns:
            Iterator over points
        """

        if self._data is not None:
            for i in self._data:
                for j in i:
                    yield j
        else:
            yield from ()


    @staticmethod
    def load(f: Union[Path, str], **kwargs) -> None:
        """Loads file content

        Depending on the filename extension, a suitable loader is
        called:

            * .p: :func:`pickle.load`
            * .npy: :func:`numpy.load`
            * None: :func:`numpy.loadtxt`
            * .xvg, .dat: :func:`numpy.loadtxt`

        Sets :attr:`data` and :attr:`shape`.

        Args:
            f: File

        Keyword Args:
            **kwargs: Passed to loader.
        """
        # add load option for dist_matrix, map_matrix

        extension = Path(f).suffix

        case_ = {
            '.p': lambda: pickle.load(
                open(f, 'rb'),
                **kwargs
                ),
            '.npy': lambda: np.load(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
            '': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
            '.xvg': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
            '.dat': lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs
                ),
             }

        return case_.get(
            extension,
            lambda: print(f"Unknown filename extension {extension}")
            )()


        self.shape_str = {**self._shape}
        if self.size > 0:
            self.shape_str['points'] = (
                sum(self.shape_str['points']),
                self.shape_str['points'][:5]
                )

            if len(self._shape['points']) > 5:
                self.shape_str['points'] += ["..."]
