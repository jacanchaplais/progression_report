---
index: [index](index.md)
title: introduction
next: [theory](theory.md)
...

# Introduction || project:progression.report.intro
* [X] GNNs generalise to arbitrary geometry  #9bd7a027
* [X] particle physics => point cloud data  #7cbeefbb
* [ ] open problems (2021-06-15 20:00)  #859bb08a
* [ ] usual babble about NNs, CNNs, _etc._  #3c640c8c
* [ ] successful applications  #990d6591
* [ ] section outline  #2d7f7a3e

Neural network models were first proposed in the ~1950s~.

* [ ] check date NNs first proposed  #82851da2

While the overhead for deep architectures was prohibitive at the time,
the last decade has seen a resurgence of interest in these techniques,
as powerful GPUs and advances in memory storage and access have enabled
applications to increasingly data intensive domains.
Of particular interest is the design of architectures
which aggregate related data points to learn useful high level features,
which are analysed for inference in the final layer of the network.

Consideration of which methods to use depends on the data topology.
Sensible choices are rewarded by fast convergence, reduced computational
and memory complexity, and improved performance at inference time.
Poor choices may result in a model which fails to provide useful predictions
regardless of available resources.
The application of convolutional neural networks (CNNs) to computer vision has
been a major domain-specific success story.
* [ ] insert impressive classification results  #dbcbb1ac
* [ ] explanation and examples of RNNs  #b419180f

see page 4 of [this paper](file:../../references/graph_track_reconstruct.pdf)
for CNN and RNN stuff references.

However, measurements within many domains of research have a more
complex notion of locality. In high energy physics research, particle
colliders are equipped with sensors that detect sprays of incident particles.
These sprays are naturally described as _point clouds_: sparsely arranged
sets of readings. The location and value of a point within this set is often
strongly correlated with other points, as they were produced by the
particles which were interacting before they hit the detector. A natural
method of describing the neighbourhood for a given measurement is then to link
the members of the set with which it is correlated. This definition of locality
forms a graph topology, and aggregating over this neighbourhood for deep
learning applications generalises the CNN approach to arbitrary spatial
structures.


