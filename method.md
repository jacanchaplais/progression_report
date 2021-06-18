---
index: [index](index.md)
title: method
prev: [theory](theory.md)
next: [results](results.md)
...

# Method || project:progression.report.method

## Lorentz group networks

In preparation to explore graph neural network designs, existing applications
to HEP were tested. This started with the Lorentz Group Network [citation],
in which my primary aim was to create a data pipeline from MadGraph to
HDF5 files that could interface with the network. This was successfully
achieved by applying cuts to the data to exclude particles with a
pseudorapidity higher than 2.5, as these would not hit the detector, and a
transverse momentum below 0.5, as this would not be detected even if incident.
Following the cuts, the data was then clustered using the anti-kT algorithm,
and the resulting jets were then tagged with identity of the particle which
produced them. Finally, the jet with the highest transverse momentum was then
selected and saved, ready to be fed into the neural network.  The command line
programs developed for this data generation automate the process end-to-end,
from generating the events via MadGraph, to the post processing, and finally
formatting the data for the given network. New HDF5 file structures can
be easily added to these programs, to enable flexibility between different
projects, and future design choices for this project.

## Interaction networks

To obtain hands on experience in developing graph networks, the interaction
network (IN) was implemented, as applied by Ju et al. for the purpose
of jet clustering. An attempt has been made to recreate their work and results.
The function of this graph network is to take the output of an entire particle
collision event, without cuts on the pseudorapidity or the transverse momentum,
and cluster the particles belonging to a boosted W boson jet. This was
compared directly against the MC truth, which enabled labelling of the final
state particles originating from the W. This task is more complex for quark
or gluon jets as, unlike the W, they are not colour neutral, _ie._ they form
_colour triplets_.
Final state particles must be colour neutral, _ie._ form _colour
singlets_, therefore, more than one of these partons from the hard process must
participate in the showering.
This is usually resolved by determining which parent parton provided the
largest transverse momentum, but by considering only the colour singlet case
of the W boson, the need for these stipulations during the early stage of this
project was avoided.

### Architecture

In the original work defining IN, the input to the network is an attributed,
directed multigraph. The nodes represent physical objects, and the edges
represent relationships between the pairs of objects they connect.
An additional matrix representing external effects on each object may be
provided alongside the graph, to represent interactions outside the system,
_eg._ with gravity.

The function to encode the nodes and edges, and the function to create
successive embeddings, are identical. In the general case of the edge update
function, the node features
of a send / receive pair, as well as any edge features between them,
are stacked in a single column vector. This process is repeated for each
send / receive pair, and the resulting column vectors are concatenated
horizontally to form
a matrix, where each column indexes one send / receive pair. This is then
fed through a MLP to obtain an updated edge embedding for the graph.

The general node update function proceeds by first aggregating the new
edge embedding over all incoming edges for each receiving node. The aggregation
function used is a summation. Each receiving node's edge sum is a column
vector, referred to as the _effect_ applied to the object the node represents.
These vectors are concatenated vertically with their respective node embeddings
as well as the external effects provided, to form a new column vector. This
is then repeated for all nodes in the graph, and the resulting column vectors
are concatenated horizontally to form a matrix, in which the columns refer to
individual nodes. Finally, this is input to a MLP to form an updated node
embedding, taking account of the internal and external interactions on the
system.

If a graph level inference is required, IN aggregates the node embeddings, by
summation over the feature vectors to form a single column vector. This is then
fed through a multilayer perceptron to form a graph embedding.

The resulting embeddings are then passed through a linear layer for
classification or regression tasks.

### Implementation

In the specific implementation of the IN for jet clustering, external effects
are not considered, and the edges are initialised without attributes. The
particles are represented as a fully connected, bidirectional graph.
Node features are given by the four momenta of the final state
particles. The truth labels for edge prediction are binary: true if
both particles should be clustered within the jet,
false otherwise. The network is then trained to find edge
embeddings from the node features. These are then run through a linear layer,
and a sigmoid activation to provide a real valued number in the interval
[0, 1], _ie._ an edge level classification score. Scores closer to 0 suggest
the edge is less likely to be within the cluster, and _vice versa_.

The MLPs are implemented as two-layer fully-connected NNs, with ReLU
activations applied to the outputs.

The cost function used is binary cross entropy, with a weighting in favour of
improving the recall of the classifier, as (without using cuts)
the background forms over 99\% of the input. Adam is used to minimise this
cost.

The authors of the jet clustering IN used TensorFlow to implement their
own graph representations and message passing. In the recreated version for
this project, the network was defined with PyTorch [@pytorch],
using PyTorch Geometric [@pytorch_geometric]
for the message passing layers and graph representations, and
PyTorch Lightning [@pytorch_lightning]
to parallelise for multiple GPUs.

### Tuning

In order to ensure appropriate hyperparameters were selected,
Ray Tune [@ray_tune]
was used to tune the model, using the HyperOpt algorithm [@hyperopt]
to manage the exploration of the search space,
and the Asynchronous Successive Halving Algorithm (ASHA) [@asha]
to stop unpromising trials early. Logarithmic search spaces within the interval
$[10^{-6}, 10^{-1}]$ were explored for the learning rate and weight decay of
Adam, and both linear and logarithmic search spaces in the interval $[1, 120]$
were used for the weighting on the positive examples in the cost. This
was done for 100 trial training runs over a maximum of 12 epochs.
The networks trained used 128 dimensional embeddings for both the nodes and
edges, as in the original jet clustering IN, but to save on memory complexity
subsequent graph layers produced 64 dimensional embeddings. To enable rapid
sampling, four trials were performed simultaneously with one GPU each, rather
than parallelising each trial over all four GPUs available on a single node
in the Alpha cluster. As the memory requirements for calculation on on a fully
connected graph of several hundred final state particles are formidable,
only one graph per GPU may be held in memory at once, so during tuning
this limited the batch size to one graph.

The optimal parameters found from these trials are summarised in the table
below.


| Learning rate      | Weight decay       | Positive weight |
|--------------------|--------------------|-----------------|
| $1 \times 10^{-4}$ | $3 \times 10^{-5}$ | $4.5$           |

Table:  Optimal parameters found during tuning, for training the network.


It is surprising that the weighting on the positive examples is small
compared with the ratio of background to signal, which is ~ $100\times$.
During training, this was the value which yielded the most similar scores
for converged recall and precision.

