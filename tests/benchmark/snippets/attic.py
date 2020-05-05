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