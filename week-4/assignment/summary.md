## q2 

A knowledge graph is a way of storing real world facts as a graph. Each node is an actual entity : a person, a place, a movie, a company — and each edge is a labelled relationship between two entities. So instead of just recording that two things are connected, it records exactly how they're connected. 

The defining feature is that the edges are typed, the label on the edge is the meaning.  The edges also have a direction, and the graph holds many different types of entity all mixed together, rather than one uniform type of node. 


## q3 — Over-smoothing vs over-squashing

Over-smoothing : is when stacking too many layers makes all the node representations converge to nearly the same vector. Each layer replaces a node with an aggregate of its neighbours, and repeating this acts like a smoothing filter applied over and over, so the differences between nodes wash out and the model can no longer tell them apart. Fixes: skip connections, Jumping Knowledge, normalisation such as PairNorm, or simply using fewer layers.

Over-squashing : is when information from distant nodes gets compressed and lost. A node's receptive field grows roughly exponentially with each added hop, but the fixed-size vector meant to carry that information does not, so signals from far away get squeezed through bottleneck edges and can't be represented. It depends on the graph's structure rather than depth alone. Fixes: rewiring to relieve bottlenecks, adding a virtual node connected to all others, global attention, or a larger hidden dimension.

The tension between them : reaching far-away information requires a larger receptive field, which means stacking more layers. But more layers cause over-smoothing, where nodes lose the distinctiveness needed to classify them, and a larger receptive field causes over-squashing, where the distant information can't fit through the bottlenecks regardless. Depth therefore can't simply be increased, which is why GNNs tend to stay shallow and rely on other techniques for long-range information.

## q5

The data is a table of students, and on the surface every row is independent. But students aren't really independent: students with similar habits, similar study hours, sleep, attendance, stress, tend to end up with similar grades. That is the kind of relationship a graph is built to capture. If you connect students who resemble each other, then a student's prediction can draw on the students most like them, instead of being judged on its own row in isolation. 

To actually build the graph, each student becomes a node and their attributes become the node features. Since there are no real edges in the data, the edges are created with a k-nearest-neighbours rule: each student is connected to the k students closest to them in feature space. The task is then node classification, predicting a student's grade band.

The honest result is that this didn't beat a plain model. A random forest using the same features and no graph scored 0.825, while the GAT scored 0.805. The reason is that the edges were manufactured from the features themselves, so they carry no information that wasn't already in the features.

