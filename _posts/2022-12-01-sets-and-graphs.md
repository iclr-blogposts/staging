---
layout: distill
title: Universality of Neural Networks on Sets vs. Graphs
description: Universal function approximation is one of the central tenets in theoretical deep learning research. It is the question of whether a specific neural network architecture is, in theory, able to approximate any function of interest. The ICLR paper “How Powerful are Graph Neural Networks?” shows that mathematically analysing the constraints of an architecture as a universal function approximator and alleviating these constraints can lead to more principled architecture choices, performance improvements, and long-term impact on the field. Specifically in the fields of learning on sets and learning on graphs, universal function approximation is a well-studied property. The two fields are closely linked because the need for permutation invariance in both cases leads to similar building blocks. However, we argue that these two fields have sometimes evolved in parallel, not fully exploiting their synergies. This post aims at bringing these two fields closer together, particularly from the perspective of universal function approximation.
date: 2022-12-01
htmlwidgets: true

# Anonymize when submitting
#authors:
#  - name: Anonymous

authors:
  - name: Fabian B. Fuchs*
    url: "https://fabianfuchsml.github.io"
    affiliations:
      name: Google DeepMind
  - name: Petar Veličković*
    url: "https://petar-v.com/"
    affiliations:
      name: (*equal contribution)

# must be the exact same name as your blogpost
bibliography: 2022-12-01-sets-and-graphs.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Sets and Graphs
  - name: Why do we care about universal function approximation?
  - name: Learning on Sets & Universality
  - name: Approximation vs. Representation
  - name: What about _graph_ representation learning?
  - name: Learning on Graphs and Universality
  - name: The Weisfeiler-Lehman Test
  - name: Broader Context and Takeaways

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Sets and Graphs

Before we dive into<d-footnote>It is important to briefly focus on declaring the *conflict of interest* we had while writing this blog. We are actively working on set and graph representation learning. Accordingly, several paragraphs of this write-up focus on papers that we have co-written. That being said, and in the context of ICLR, we declare that the majority of the ICLR papers referenced in this blog post do _not_ present a conflict of interest for us. Hence, we believe we have, to the best of our efforts, provided an objective and impartial view of learning universal representations over graphs and sets.</d-footnote> universal function approximation, let's start with the basics. What do we mean by learning on set- or graph-based data? In both cases, we assume no ordering (we will more formally describe this at the end of this section as the task being permutation _invariant_ or _equivariant_). A graph is typically thought of as a set of nodes with edges between the nodes. A set doesn't have edges, it just has the nodes, although we often don't call them nodes, but rather set elements. Both the nodes and the edges (in the case of graphs) can have feature vectors attached to them. The figure below (originally from Wagstaff et al. 2021<d-cite key="wagstaff21"></d-cite>) visualises this relationship:

{% include figure.html path="assets/img/2022-12-01-sets-and-graphs/graphsuniv_graphsandsets.png" class="img-fluid" %}

Examples of machine learning tasks on this type of data include 3D point cloud classification (a function mapping a set of coordinates to an object class) and molecular property prediction (a function mapping a molecular graph to, e.g., a free energy value).

So, what are invariance and equivariance? Both concepts describe how the output of a function (or task) changes under a transformation of the input. Transformation can mean different things, but we restrict ourselves to permutations here for simplicity. A function $$f$$ is permutation *invariant* if the output does not change as the inputs are permuted. The left-hand side of the following figure below visualises that concept:


{% include figure.html path="assets/img/2022-12-01-sets-and-graphs/graphsuniv_permutations.png" class="img-fluid" %}

The right-hand side depicts permutation _equivariance:_ changing the order of the input implies a change in the order of the output (but the values themselves remain unchanged).

Tasks (or functions) defined on sets and graphs are typically permutation invariant or equivariant. This symmetry is often incorporated into the neural network architecture, as we will see in examples below. It is exactly the incorporation of the symmetry that makes the question of universalilty so interesting: is the network (theoretically) able to model all permutation invariant (or equivariant) functions on this data?


## Why do we care about universal function approximation?

First of all, why do we need to be able to approximate all functions? After all, having _one_ function that performs well on the train set and generalises to the test set is all we need in most cases. Well, the issue is that we have no idea what such a function looks like, otherwise we would implement it directly and wouldn't need to train a neural network. Hence, the network not being a universal function approximator *may* hurt its performance.
<!-- So the logic is, we don't want to restrict the network unless the restrictions only refer to functions we know that we don't care about. -->

Graph Isomorphism Networks (GINs) by Xu et al.<d-cite key="GIN"></d-cite>) provide the quintessential example of the merit of universality research. The authors first realised that it is possible to mathematically describe all functions that can be computed by graph neural networks relying on message passing between immediate neighbours, over graphs with discrete-valued features. They then analysed Graph Convolutional Networks (a very popular class of graph neural networks by Kipf and Welling 2016 [3]), and pointed out that GCNs are not capable of expressing all of these functions — that is, they are not universal. Guided by their analysis, the authors then created the GIN, which was provably capable of expressing all possible such functions and achieved significantly better empirical results.
<!-- Graph Isomorphism Networks (GINs) by Xu et al.<d-cite key="GIN"></d-cite>) provide the quintessential example for the merit of universality research: the authors analysed Graph Convolutional Networks (a very popular class of graph neural networks by Kipf et al. 2016<d-cite key="GCN"></d-cite>), pointed out that GCNs are not universal, created a variation of the algorithm that *is* universal (or at least closer to), and achieved better results. -->
<!-- So, in this case, the non-universality of the GCNs really did hurt their performance. -->

However, this is not always the case. Sometimes, architecture changes motivated by universal function approximation arguments lead to *worse* results. Even in such unfortunate cases, however, we argue that thinking about universality is no waste of time. Firstly, it brings structure into the literature and into the wide range of models available. We need to group approaches together to see the similarities and differences. Universality research can and has served as a helpful tool for that.

Moreover, proving that a certain architecture is or is not universal is an inherently interesting task and teaches us mathematical thinking and argumentation. In a deep learning world, where there is a general sense of randomness and magic in building high-performing neural networks and where it’s hard to interpret what’s going on, one might argue that an additional mathematical analysis is probably good for the balance, even if it turns out to not always directly result in better performance. 



## Learning on Sets & Universality

To prove universal function approximation<d-footnote>There is actually a nuanced distinction between *approximation* and *representation*, which we will glance over for now but discuss in the next section.</d-footnote>, we typically make two assumptions: 
1) the MLP components of the neural networks are arbitrarily large.
2) the functions that we want to be able to learn are continuous on $$\mathbb{R}$$. Continuity for a function $$f(x)$$ mapping from $$\mathbb{R}$$ to $$\mathbb{R}$$ implies that for all $$x_0$$ in the domain of $$f$$ and all $$\epsilon > 0, \epsilon \in R$$, there exists a $$\delta > 0, \delta \in R$$ such that $$|x - x_0| < \delta$$ implies $$|f(x) - < f(x_0)| < \epsilon$$ if $$x$$ is in the domain of $$f$$. 

The first part says: any concrete implementation of a 'universal' network architecture might not be able to learn the function of interest, but, if you make it [bigger](https://i.redd.it/n9fgba8b0qr01.png), eventually it will---and that is *guaranteed*<d-footnote>Conversely, if the network is provably non-universal (like Graph Convolutional Networks), then there are functions it can *never* learn, no matter how many layers you stack.</d-footnote>. The second part is a non-intuitive mathematical technicality we will leave uncommented for now and get back to later (because it's actually a really interesting and important technicality).



One of the seminal papers discussing both permutation invariant neural networks and universal function approximation was Deep Sets by Zaheer et al. in 2017<d-cite key="Zaheer2017"></d-cite>. The idea is simple: apply the same neural network $$\phi$$ to several inputs, sum up their results, and apply a final neural network $$\rho$$.<d-footnote>Figure from Wagstaff et al. 2021.</d-footnote>

{% include figure.html path="assets/img/2022-12-01-sets-and-graphs/graphsuniv_deepsets.png" class="img-fluid" %}


Because the sum operation is permutation invariant, the final output is invariant with respect to the ordering of the inputs. In other words, the sum quite obviously restricts the space of learnable functions to permutation invariant ones. The question is, can a neural network with this architecture, in principle, learn _all_ (continuous) permutation invariant functions? Perhaps surprisingly, the authors show that all functions can indeed be represented with this architecture. The idea is a form of binary bit-encoding in the output of $$\phi$$, which we will call the _latent space_ from here on. Concretely, they argue that there is a bijective mapping from rational to natural numbers. Assuming that each input is a rational number, they first map each rational number $$x$$ to a natural number $$c(x)$$, and then each natural number to $$\phi(x) = 4^{-c(x)}$$. It is now easy to see that $$\sum_i \phi(x_i) \neq \sum_i \phi(y_i)$$ unless the finite sets $$ \\{ x_0, x_1, ... \\} $$ and $$\\{y_0, y_1, ...\\}$$ are the same. Now that we uniquely encoded each input, a universal decoder can map this to any output we want. This concludes the proof that the Deep Sets architecture is, in theory, a universal function approximator, despite its simplicity.

However, there is an issue with this proof: it builds on the assumption that the MLP components themselves are universal function approximators, in the limit of infinite width. However, the universal function approximation theorem says that this is the case only for continuous functions, where continuity is defined on the real numbers. That continuity is important is sort of intuitive: continuity means that a small change in the input implies a small change in the output. And because the building blocks of neural networks (specifically linear combinations and non-linearities) are continuous, it makes sense that the overall function we want the network to learn should be continuous.

But why continuity on the real numbers? Because continuity on the rational numbers is not a very useful property as shown in Wagstaff et al. 2019<d-cite key="wagstaff19"></d-cite>. The mapping we described above is clearly highly discontinuous, and anyone could attest that it is completely unrealistic to assume that a neural network could learn such a function. That doesn't mean all is lost. Wagstaff et al. show that the Deep Sets architecture is still a universal function approximator when requiring continuity, but only if the latent space (the range of $$\phi$$) has a dimensionality at least as large as the number of inputs, which is an important restriction.


What about more complicated architectures? Murphy et al.<d-cite key="Janossy"></d-cite> generalise the idea of Deep Sets to applying networks to all possible $$k$$-tuples of inputs, where $$k=1$$ recovers the Deep Sets case. This can be seen as unifying other architecture classes such as self-attention. However, this is not known to alleviate the constraint on the latent space mentioned above, as explained in Wagstaff et al. 2021<d-cite key="wagstaff21"></d-cite>.


## Approximation vs. Representation

For simplicity, we have so far deliberately glanced over the distinction between function approximation and representation, but we will rectify this now. The Deep Sets architecture from the previous section can be written as:

$$\rho (\sum \phi_i(x_i))$$

If we forget about $$\rho$$ and $$\phi$$ being implemented as neural networks for a second and just think of them as general functions, it turns out that any continuous permutation invariant function can be _represented_ in the above way. The word _represented_ implies that it's exact, without an approximation error, not even an arbitrarily small one. As such, Zaheer et al. 2017<d-cite key="Zaheer2017"></d-cite> and Wagstaff et al. 2019<d-cite key="wagstaff19"></d-cite> study universal function *representation*, not the softer criterion of *approximation*. However, once we assume $$\rho$$ and $$\phi$$ are being implemented as neural networks, it is an approximation. Hence, it makes sense to call Deep Sets a universal function *approximator* for continuous functions on sets. There is a catch here, though. If we are satisfied with approximation in the components $$\phi$$ and $$\rho$$, we might as well be satisfied with approximations in other places as well. A question one could ask is "how large does the latent space have to be in order to keep the errors small?". This is unsurprisingly a much harder question to answer, but Wagstaff et al. 2021<d-cite key="wagstaff21"></d-cite> find that the result is largely the same: the latent space much have a dimensionality at least as large as the number of inputs.




## What about _graph_ representation learning?

So, this was universality in the context of machine learning on sets, but what about graphs? Interestingly, the graph representation learning community experienced a near-identical journey, evolving entirely in parallel! Perhaps this observation comes as little surprise: to meaningfully propagate information in a graph neural network (GNN), a local, permutation invariant operation is commonplace. 

<!-- Specifically, a GNN typically operates by computing representations (_"messages"_) sent from each node to its neighbours, followed by an _aggregation function_ which, for every node, combines all of its incoming messages in a way that is _invariant to permutations_.  -->

 Specifically, a GNN typically operates by computing representations (_"messages"_) sent from each node to its neighbours, using a _message function_<d-footnote>Here, for the purpose of clarity, we assume that the message function $\psi$ only takes into account the features of the sender and receiver nodes. It is of course possible to have additional relevant features in the graph that $\psi$ could use, for example, there could be features on the edge $i\rightarrow j$, as is often the case, e.g., in molecular graphs. Such cases can usually be resolved by inserting these features as additional inputs to $\psi$.</d-footnote>, $\psi : \mathbb{R}^k \times \mathbb{R}^k\rightarrow\mathbb{R}^l$:

$$\mathbf{m}_{ij} = \psi(\mathbf{x}_{i}, \mathbf{x}_{j})$$

where $$\mathbf{x}_{i}$$ are the features of node $i$. This is followed by an _aggregation function_ which, for every node, combines all of its incoming messages in a way that is invariant to permutations:

$$\mathbf{h}_{i} = \phi\left(\mathbf{x}_{i}, \bigoplus_{j\in\mathcal{N}_{i}} \mathbf{m}_{ji}\right)$$

where $$\mathcal{N}_i$$ is the set of all nodes neighbouring $i$, and $$\phi : \mathbb{R}^k\times\mathbb{R}^l\rightarrow\mathbb{R}^m$$ is an _update function_, updating the representation of each node $$i$$ from $$\mathbf{x}_{i}$$ to $$\mathbf{h}_{i}$$.

Opinions are still divided on whether _every_ permutation equivariant GNN can be expressed with such pairwise messaging, with a recent position paper by Veličković<d-cite key="Velickovic22"></d-cite> claiming they **can**. Regardless of which way the debate goes in the future, aggregating messages over 1-hop neighbours gives rise to a highly elegant implementation of GNNs which is likely here to stay. This comes with very solid community backing, with [PyG](https://www.pyg.org/)---one of the most popular GNN frameworks---[recently making aggregators a "first-class citizen"](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.1.0) in their GNN pipelining.

Therefore, to build a GNN, it suffices to build a _permutation-invariant, local_ layer which combines data coming from each node's neighbours. This feels nearly identical to our previous discussion; what's changed, really? Well, we need to take care of one seemingly minor detail: it is possible for **two or more neighbours to send _exactly the same message_**. The theoretical framework of Deep Sets and/or Wagstaff et al. wouldn't entirely suffice in this case, as they assumed a _set_ input, whereas now we have a _multiset_ (a set where some elements might be repeated)..






## Learning on Graphs and Universality

Several influential GNN papers were able to overcome this limitation. The first key development came from the _graph isomorphism network_ (**GIN**)<d-cite key="GIN"></d-cite>. GIN is an elegant example of how, over countable features, the maximally-powerful GNN<d-footnote>That is, a GNN that is capable of expressing all possible functions that can be described using several iterations of message passing between one-hop neighbours in a graph.</d-footnote> can be built up using similar ideas as in Deep Sets; so long as the local layer we use is _injective_<d-footnote>Injectivity of a function means that two different inputs always yield two different outputs. In other words, if you evaluate the function twice and the output is the same both times, you know that the input must have been the same, too.</d-footnote> over multisets. Similarly to before, we must choose our encoder $$\phi$$ and aggregator $$\bigoplus$$, such that $$\bigoplus\limits_i \phi(x_i) \neq \bigoplus\limits_i \phi(y_i)$$ unless the finite _multisets_ $\\{  \mkern-4mu \\{x_0, x_1, ...\\} \mkern-4mu \\}$ and $\\{\mkern-4mu\\{y_0, y_1, ...\\} \mkern-4mu \\}$ are the same ($$x_i, y_i\in\mathbb{Q}$$).

In the multiset case, the framework from Deep Sets induces an additional constraint over $$\bigoplus$$---it needs to preserve the _cardinality_ information about the repeated elements in a multiset. This immediately implies that some choices of $$\bigoplus$$, such as $$\max$$ or averaging, will not yield maximally powerful GNNs.

For example, consider the multisets $\\{\mkern-4mu\\{1, 1, 2, 2\\} \mkern-4mu \\}$ and $\\{\mkern-4mu\\{1, 2\\}\mkern-4mu\\}$. As we assume the features to be countable, we specify the numbers as _one-hot_ integers; that is, $$1 = [1\ \ 0]$$ and $$2=[0\ \ 1]$$. The maximum of these features, taken over the multiset, is $$[1\ \ 1]$$, and their average is $$\left[\frac{1}{2}\ \ \frac{1}{2}\right]$$. This is the case for both of these multisets, meaning that both maximising and averaging are _incapable_ of telling them apart.

Summations $$\left(\bigoplus=\sum\right)$$, however, are an example of a suitable injective operator.

Very similarly to the analysis from Wagstaff et al. in the domain of sets, a similar extension in the domain of graphs came through the work on [_principal neighbourhood aggregation_](**PNA**) by Corso, Cavalleri et al.<d-cite key="Corso"></d-cite>. We already discussed why it is a good idea to focus on features coming from $$\mathbb{R}$$ rather than $$\mathbb{Q}$$---the universal approximation theorem only applies to functions that are continuous on $$\mathbb{R}$$. However, it turns out that, when we let $$x_i, y_i\in\mathbb{R}$$, it is easily possible to construct neighbourhood multisets for which setting $$\bigoplus=\sum$$ would **not** preserve injectivity: 

{% include figure.html path="assets/img/2022-12-01-sets-and-graphs/graphsuniv_examples.png" class="img-fluid" %}

In fact, PNA itself is based on a proof that it is _impossible_ to build an injective function over multisets with real-valued features using _any_ **single** aggregator. In general, for an injective function over $$n$$ neighbours, we need _at least_ $$n$$ aggregation functions (applied in parallel). PNA then builds an empirically powerful aggregator combination, leveraging this insight while trying to preserve numerical stability.

Note that there is an apparent **similarity** between these results and the ones from Wagstaff et al. 2019<d-cite key="wagstaff19"></d-cite> . Wagstaff et al. show that, over real-valued sets of $$n$$ elements, it is necessary to have an encoder representation _width_ of at least $$n$$. Corso, Cavalleri et al. show that, over real-valued multisets of $$n$$ elements, it is necessary to aggregate them with at least $$n$$ aggregators.

There are also major differences between the two analyses: Wagstaff et al. 2019<d-cite key="wagstaff19"></d-cite> assume the sum as an aggregator, whereas Corso et al.<d-cite key="Corso"></d-cite> consider arbitrary aggregation functions. They also use different language: number of aggregators vs. dimensionality of the latent space, although the two are equivalent. Ultimately, the restriction to sums makes the sufficiency proof (the neural network _is_ universal for num latents $$\geq$$ num inputs) for Wagstaff et al. more complicated, which uses a sum-of-power mapping. Corso et al., on the other hand, simply use an aggregator that extracts the $$i$$th-smallest input element. The necessity proof (the neural network _is not_ universal for num latents $$<$$ num inputs), on the other hand, is more complex for Corso et al. and uses the Borsuk–Ulam theorem, because all possible aggregation functions have to be taken into account. Remarkably, despite the different starting conditions, both proofs arrive at the exact same result: for a universal neural network, you need as many aggregators/latents as you have inputs.

In other words, it appears that potent processing of real-valued collections necessitates representational capacity proportional to the collection’s size, in order to guarantee injectivity. Discovering this correspondence is actually what brought the two of us together to publish this blog post in the first place.

We have now established what is necessary to create a maximally-powerful GNN over both _countable_ and _uncountable_ input features. So, _how powerful are they_, exactly?

## The Weisfeiler-Lehman Test
While GNNs are often a powerful tool for processing graph data in the real world, they also won’t solve _all_ tasks specified on a graph accurately! As a simple counterexample, consider any NP-hard problem, such as the Travelling Salesperson Problem. If we had a fixed-depth GNN that perfectly solves such a problem, we would have shown P=NP! Expectedly, not all GNNs will be equally good at solving various problems, and we may be highly interested in characterising their _expressive power_.

The canonical example for characterising expressive power is _deciding graph isomorphism_; that is, can our 
GNN distinguish two non-isomorphic graphs? Specifically, if our GNN is capable of computing graph-level 
representations $$\mathbf{h}_{\mathcal{G}}$$, we are interested whether $$\mathbf{h}_{\mathcal{G_{1}}} \neq\mathbf{h}_{\mathcal{G_{2}}}$$ for non-isomorphic graphs $$\mathcal{G}_{1}$$ and $$\mathcal{G}_{2}$$. If we cannot attach different representations to these two graphs, any kind of task requiring us to classify them differently is _hopeless_! This motivates assessing the power of GNNs by which graphs they are able to _distinguish_.

A typical way in which this is formalised is by using the _Weisfeiler-Lehman_ (**WL**) graph isomorphism test. To formalise this, we will study a popular algorithm for approximately deciding graph isomorphism.

The WL algorithm featurises a graph $$\mathcal{G}=(\mathcal{V},\mathcal{E})$$ as follows. First, we set the representation of each node $$i\in\mathcal{V}$$ as $$x_i^{(0)} = 1$$. Then, it proceeds as follows:
1. Let $\mathcal{X}_i^{(t+1)} = \\{\mkern-4mu\\{x_j^{(t)} :(i,j)\in\mathcal{E}\\}\mkern-4mu\\}$ be the multiset of features of all neighbours of $$i$$.
2. Then, let $$x_i^{(t+1)}=\sum\limits_{y_j\in\mathcal{X}_i^{(t+1)}}\phi(y_j)$$, where $$\phi : \mathbb{Q}\rightarrow\mathbb{Q}$$ is an _injective_ hash function.

This process continues as long as the _histogram_ of $$x_i^{(t)}$$ changes---initially, all nodes have the same representation. As steps 1--2 are iterated, certain $$x_i^{(t)}$$ values may become different. Finally, the WL test checks whether two graphs are (possibly) isomorphic by checking whether their histograms have the same (sorted) shape upon convergence.

While remarkably simple, the WL test can accurately distinguish most graphs of real-world interest. It does have some rather painful failure modes, though; for example, it cannot distinguish a 6-cycle from two triangles!

{% include figure.html path="assets/img/2022-12-01-sets-and-graphs/graphsuniv_wlfail.png" class="img-fluid" %}

This is because, locally, _all nodes look the same_ in these two graphs, and the histogram never changes.

The key behind the power of the WL test is the _injectivity_ of the hash function $$\phi$$---it may be interpreted as assigning each node a different _colour_ if it has a different _local context_. Similarly, we saw that GNNs are maximally powerful when their propagation models are _injective_. It should come as little surprise then that, in terms of distinguishing graph structures over _countable_ input features, GNNs can **never be more powerful than the WL test**! And, in fact, this level of power is achieved _exactly_ when the aggregator is injective. This fact was first discovered by Morris et al.<d-cite key="Morris"></d-cite>, and reinterpreted from the perspective of multiset aggregation by the GIN paper.

While the WL connection has certainly spurred a vast amount of works on improving GNN expressivity, it is also worth recalling the initial assumption: $$x_i^{(0)} = 1$$. That is, we assume that the input node features are _completely uninformative_! Very often, this is not a good idea! It can be proven that even placing _random numbers_ in the nodes can yield to a provable improvement in expressive power (Sato et al.<d-cite key="Sato"></d-cite>). Further, many recent works (Loukas et al.<d-cite key="Loukas"></d-cite>); Kanatsoulis and Ribeiro<d-cite key="Ribeiro"></d-cite> make it very explicit that, if we allow GNNs access to "appropriate" input features, this leads to a vast improvement in their expressive power. All of these models hence surpass the 1-WL test. There is now a significant body of recent research to improve GNNs beyond the 1-WL test by giving them access to features or structures they wouldn't otherwise be capable of computing. The broad strategies for doing so, beyond the just-discussed feature augmentation, include rewiring the graph, and explicit message passing over _substructures_ in the graph. Veličković<d-cite key="Velickovic22"></d-cite> provides a bird's eye summary of these recent developments.

Even beyond the limitation of the uninformative input features, recent influential works (published at ICLR'22 and '23 as orals) have demonstrated that the WL framework itself is worth extending. Geerts and Reutter<d-cite key="Geerts"></d-cite> demonstrate clear theoretical value to expressing GNN computations using a _tensor language_ (TL), allowing for drawing significant connections to _color refinement_ algorithms. And Zhang et al.<d-cite key="Zhang"></d-cite> demonstrate that the WL framework may be _weak_ in terms of its architectural distinguishing power, showing that many higher-order GNNs that surpass the limitations of the 1-WL test are in fact still incapable of computing many standard polynomial-time-computable properties over graphs, such as ones relating to the graph's _biconnected components_.

Lastly, linking back to our central discussion, we argue that focusing the theoretical analysis only on discrete features may not lead to highly learnable target mappings. From the perspective of the WL test (and basically any discrete-valued procedure), the models presented in Deep Sets and PNA are no more powerful than 1-WL. However, moving into continuous feature support, PNA is indeed more powerful at distinguishing graphs than models like GIN.

## Broader Context and Takeaways

It is no coincidence that many of the current universality discussions within machine learning are happening inside communities that build networks that exploit symmetries (in our examples, the symmetry was always permutation invariance/equivariance, but the following argument equally applies to, e.g., rotational symmetries): exploiting symmetries with a neural network architecture is tantamount to limiting the space of functions that can be learned. This naturally raises the question of _how much_ the space of learnable function has been limited. In other words: for the space of functions observing a specific symmmetry, is the neural network (still) a universal function approximator? This does not imply, however, that universality isn't interesting in other fields, too: e.g., the fact that self-attention (popularised by natural language processing) is a universal approximator for functions on sets is an interesting property that gives its design more context. The (once) ubiquitous usage of the convolutional layer seems less surprising when knowing that it is the most general<d-footnote>In fact, it is also the only such linear layer because simpler and less expressive translation equivariant linear layers (e.g. point-wise linears) can be seen as special cases of a convolutional layer.</d-footnote> linear layer that observes translation equivariance<d-cite key="Cohen"></d-cite>.

In this blog post, we aimed at explaining most of the key concepts of universal function approximation for set and graph-based machine learning: invariance and equivariance, sets and multisets, representation vs. approximation, injectivity, Deep Sets, GINs, WL-tests, and the motivation for universality research itself. We hope that we provided some insights into the similarities and differences of universality research on graphs and sets, and maybe even food for thought leading to future research on this intersection. We also acknowledge that this is a theoretical topic and that none of these proofs can ultimately predict how well a 'universal' neural network will perform on a specific task in the real world. However, even in the worst-case scenario, where theoretical universality properties are completely uncorrelated (or inversely correlated?) with real-world performance, we still hope that the thoughts and concepts of this post add a bit of additional structure to the multifaceted zoo of neural network architectures for sets and graphs.



