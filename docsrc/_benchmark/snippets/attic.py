# CLuster cleanup

clusters_no_noise = {y: self._clusterdict[y] for y in self._clusterdict if y != 0}

too_small = [
    self._clusterdict.pop(y)
    for y in [x[0] for x in clusters_no_noise.items() if len(x[1]) <= member_cutoff]
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
    largest = (
        len(
            self._clusterdict[
                1 + np.argmax([len(x) for x in clusters_no_noise.values()])
            ]
        )
        / self.data.shape_str["points"][0]
    )

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
        [
            len(np.where(self._labels == max(self._labels))[1])
            / self.data.shape_str["points"][0]
        ],
        [len(np.where(self._labels == 0)[0]) / self.data.shape_str["points"][0]],
        [None],
    ],
)

if v:
    print("\n" + "-" * 72)
    print(
        cresult[list(self.record._fields)[:-1]].to_string(
            na_rep="None",
            index=False,
            line_width=80,
            header=[
                "  #points  ",
                "  R  ",
                "  N  ",
                "  M  ",
                "  max  ",
                "  #clusters  ",
                "  %largest  ",
                "  %noise  ",
            ],
            justify="center",
        )
    )
    print("-" * 72)

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

            for part in range(self._shape["parts"]):
                part_endpoint = part_startpoint + self._shape["points"][part] - 1

                cluster_data.append(
                    self._data[part][
                        cpoints[
                            np.where(
                                (cpoints >= part_startpoint)
                                & (cpoints <= part_endpoint)
                            )[0]
                        ]
                        - part_startpoint
                    ]
                )
                part_startpoint = np.copy(part_endpoint)
                part_startpoint += 1

            self._children[label].alias = f"child No. {label}"
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
            ".p": lambda: pickle.load(open(f, "rb"), **kwargs),
            ".npy": lambda: np.load(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs,
            ),
            "": lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs,
            ),
            ".xvg": lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs,
            ),
            ".dat": lambda: np.loadtxt(
                f,
                # dtype=float_precision_map[float_precision],
                **kwargs,
            ),
        }

        return case_.get(
            extension, lambda: print(f"Unknown filename extension {extension}")
        )()

        self.shape_str = {**self._shape}
        if self.size > 0:
            self.shape_str["points"] = (
                sum(self.shape_str["points"]),
                self.shape_str["points"][:5],
            )

            if len(self._shape["points"]) > 5:
                self.shape_str["points"] += ["..."]

        #### Predict

        len_ = len(_test)  # self.data.points.shape[0]

        # TODO: Decouple memorize?
        if purge or (clusters is None):
            self.__test_labels = np.zeros(len_).astype(int)
            self.__memory_assigned = np.ones(len_).astype(bool)
            if clusters is None:
                clusters = list(self.data.points.clusterdict.keys())

        else:
            if self.__memory_assigned is None:
                self.__memory_assigned = np.ones(len_).astype(bool)

            if self.__test_labels is None:
                self.__test_labels = np.zeros(len_).astype(int)

            for cluster in clusters:
                self.__memory_assigned[self.__test_labels == cluster] = True
                self.__test_labels[self.__test_labels == cluster] = 0

            _test = _test[self.__memory_assigned]
            # _map = self.__map_matrix[self.__memory_assigned]

        progress = not progress

        if behaviour == "on-the-fly":
            if method == "plain":
                # TODO: Store a vstacked version in the first place
                _train = np.vstack(self.__train)

                r = radius_cutoff ** 2

                _test_labels = []

                for candidate in tqdm.tqdm(
                    _test,
                    desc="Predicting",
                    disable=progress,
                    unit="Points",
                    unit_scale=True,
                    bar_format="%s{l_bar}%s{bar}%s{r_bar}"
                    % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET),
                ):
                    _test_labels.append(0)
                    neighbours = self.get_neighbours(candidate, _train, r)

                    # TODO: Decouple this reduction if clusters is None
                    try:
                        neighbours = neighbours[
                            np.isin(self.__train_labels[neighbours], clusters)
                        ]
                    except IndexError:
                        pass
                    else:
                        for neighbour in neighbours:
                            neighbour_neighbours = self.get_neighbours(
                                _train[neighbour], _train, r
                            )

                            # TODO: Decouple this reduction if clusters is None
                            try:
                                neighbour_neighbours = neighbour_neighbours[
                                    np.isin(
                                        self.__train_labels[neighbour_neighbours],
                                        clusters,
                                    )
                                ]
                            except IndexError:
                                pass
                            else:
                                if self.check_similarity_array(
                                    neighbours, neighbour_neighbours, cnn_cutoff
                                ):
                                    _test_labels[-1] = self.__train_labels[neighbour]
                                    # break after first match
                                    break
            else:
                raise ValueError()

        elif behaviour == "lookup":
            _map = self.__map_matrix[self.__memory_assigned]
            _test_labels = []

            for candidate in tqdm.tqdm(
                range(len(_test)),
                desc="Predicting",
                disable=progress,
                unit="Points",
                unit_scale=True,
                bar_format="%s{l_bar}%s{bar}%s{r_bar}"
                % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET),
            ):

                _test_labels.append(0)
                neighbours = np.where(_map[candidate] < radius_cutoff)[0]

                # TODO: Decouple this reduction if clusters is None
                try:
                    neighbours = neighbours[
                        np.isin(self.__train_labels[neighbours], clusters)
                    ]
                except IndexError:
                    pass
                else:
                    for neighbour in neighbours:
                        neighbour_neighbours = np.where(
                            (self.__train_dist_matrix[neighbour] < radius_cutoff)
                            & (self.__train_dist_matrix[neighbour] > 0)
                        )[0]

                        try:
                            # TODO: Decouple this reduction if clusters is None
                            neighbour_neighbours = neighbour_neighbours[
                                np.isin(
                                    self.__train_labels[neighbour_neighbours], clusters
                                )
                            ]
                        except IndexError:
                            pass
                        else:
                            if self.check_similarity_array(
                                neighbours, neighbour_neighbours, cnn_cutoff
                            ):
                                _test_labels[-1] = self.__train_labels[neighbour]
                                # break after first match

                                break

        elif behaviour == "tree":
            if self.__train_tree is None:
                raise LookupError(
                    "No search tree build for train data. Use CNN.kdtree(mode='train, **kwargs) first."
                )

            _train = np.vstack(self.__train)

            _test_labels = []

            for candidate in tqdm.tqdm(
                _test,
                desc="Predicting",
                disable=progress,
                unit="Points",
                unit_scale=True,
                bar_format="%s{l_bar}%s{bar}%s{r_bar}"
                % (colorama.Style.BRIGHT, colorama.Fore.BLUE, colorama.Fore.RESET),
            ):
                _test_labels.append(0)
                neighbours = np.asarray(
                    self.__train_tree.query_ball_point(
                        candidate, radius_cutoff, **kwargs
                    )
                )

                # TODO: Decouple this reduction if clusters is None
                try:
                    neighbours = neighbours[
                        np.isin(self.__train_labels[neighbours], clusters)
                    ]
                except IndexError:
                    pass
                else:
                    for neighbour in neighbours:
                        neighbour_neighbours = np.asarray(
                            self.__train_tree.query_ball_point(
                                _train[neighbour], radius_cutoff, **kwargs
                            )
                        )
                        try:
                            # TODO: Decouple this reduction if clusters is None
                            neighbour_neighbours = neighbour_neighbours[
                                np.isin(
                                    self.__train_labels[neighbour_neighbours], clusters
                                )
                            ]
                        except IndexError:
                            pass
                        else:
                            if self.check_similarity_array(
                                neighbours, neighbour_neighbours, cnn_cutoff
                            ):
                                _test_labels[-1] = self.__train_labels[neighbour]
                                # break after first match
                                break
        else:
            raise ValueError(
                f'Behaviour "{behaviour}" not known. Must be one of "on-the-fly", "lookup" or "tree"'
            )

        self.__test_labels[self.__memory_assigned] = _test_labels
        self.labels2dict(mode="test")

        if memorize:
            self.__memory_assigned[np.where(self.__test_labels > 0)[0]] = False
