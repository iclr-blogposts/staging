---
layout: distill
title: How much meta-learning is in image-to-image translation?
description: ...in which we find a connection between meta-learning literature and a paper studying how well CNNs deal with nuisance transforms in a class-imbalanced setting. Closer inspection reveals a surprising amount of similarity - from meta-information to loss functions. This implies that the current conception of meta-learning might be too narrow.
date: 2022-12-01
htmlwidgets: true

authors:
  - name: Maximilian Eißler
    url: "https://www.linkedin.com/in/maximilian-eißler-b51b9213b/"
    affiliations:
      name: TU Berlin
  - name: Thomas Goerttler
    url: "https://scholar.google.de/citations?user=ppQIwpIAAAAJ&hl=de"
    affiliations:
      name: NI, TU Berlin
  - name: Klaus Obermayer
    url: "https://www.tu.berlin/ni/"
    affiliations:
      name: NI, TU Berlin 

# must be the exact same name as your blogpost
bibliography: 2022-12-01-how-much-meta-learning-is-in-image-to-image-translation.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: A closer look at the experiment
  - name: How is this a meta-learning experiment? 
  - name: Generative Invariance Transfer
  - name: How much meta-learning is in MUNIT?
    subsections:
      - name: "Part 1: The task-centered view"
      - name: "Part 2: The bi-level programming view"
      - name: "Now, does MUNIT meta-learn?"
  - name: Implications
  - name: Key Takeaways
---

At the last ICLR conference, Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> presented work showing that CNNs do not transfer information between classes of a classification task. 

- Allan Zhou, Fahim Tajwar, Alexander Robey, Tom Knowles, George J. Pappas, Hamed Hassani, Chelsea Finn [ICLR, 2022] Do Deep Networks Transfer Invariances Across Classes?<d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>

Here is a quick summary of their findings: 
If we train a Convolutional Neural Net (CNN) to classify animals on a set of randomly brightened and darkened images of cats and dogs, it will learn to ignore the scene's brightness. We say that the CNN learned that classification is **invariant** to the **nuisance transformation** of randomly changing the brightness of an image. We now add a set of leopards to the training data, but fewer examples of them (they are hard to photograph) than we have cats and dogs. However, we keep using the same random transformations. The training set thus becomes **class-imbalanced**.

We might expect a sophisticated learner to look at the entire dataset, recognize the random brightness modifications across all species of animal and henceforth ignore brightness when making predictions. If this applied to our experiment, the CNN would be similarly good at ignoring lighting variations on all animals. Furthermore, we would expect the CNN to become more competent at ignoring lighting variations in proportion to **the total amount of images**, irrespective of which animal they depict. 

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/CONCEPTUAL_DIAGRAM.svg" class="img-fluid" %}

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> show that a CNN does not behave like this: When using a CNN on a **class-imbalanced** classification task with random nuisance transformations, the CNNs invariance to the transformation is proportional to the size of the training set **for each class**. This finding suggests CNNs don't **transfer invariance** between classes when learning such a classification task.

However, there is a solution: Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use an Image-to-Image translation architecture called MUNIT<d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite> to learn the transformations and generate additional data from which the CNN can learn the invariance separately for each class. Thus, the invariance to nuisance transformations is transferred **generatively**. They call this method **Generative Invariance Transfer (GIT)**.

**So why is this an interesting result?**

In the field of machine learning many have dreamed for a long time<d-cite key="schmidhuber:1987:srl"></d-cite><d-cite key="DBLP:books/sp/98/ThrunP98"></d-cite> of a learner that, having learned a number of tasks can adapt to new tasks with little to no extra training - a learner that has learned to learn, a meta-learner. Yet, specialized meta-learners <d-cite key="DBLP:conf/icml/FinnAL17"></d-cite><d-cite key="NIPS2017_cb8da676"></d-cite><d-cite key="NIPS2016_90e13578"></d-cite><d-cite key="sung2018learning"></d-cite>  struggled to outperform baseline methods<d-cite key="DBLP:journals/corr/abs-2104-02638"></d-cite><d-cite key="DBLP:journals/corr/abs-1904-04232"></d-cite>, arguably due to high computational requirements<d-cite key="nichol2018first"></d-cite> and few large scale datasets<d-cite key="triantafillou2019meta"></d-cite>. We believe this to be caused by a too-narrow conception of what constitutes meta-learning. We argue that:

- In contradiction to recent definitions of meta-learning, the experiment described above is a meta-learning experiment.
- MUNIT is related to contemporary meta-learning methods and a meta-learner.
- These two findings point to a too-narrow conception of meta-learning in the recent literature. A wider conception based on mutual information could lead to interesting future work.

Before we proceed to the main post, let's clarify some definitions. If you are already familiar with the subject, you may skip this part. If you have only a vague notion of contemporary meta-learning you will be able to follow the article anyway. However, if you want to know more, [here](https://interactive-maml.github.io/) is a gentle introduction to MAML, one of the most popular methods.

<details>
  <summary><b> Definition: Class-Imbalanced Classification</b></summary>
  <br/>
  <p>
     In many real-world classification datasets, the number of examples for each class varies. <b>Class-imbalanced classification</b> refers to classification on datasets where the frequencies of class labels vary significantly. 
  </p>
  <p>
    It is generally more difficult for a neural network to learn to classify classes with fewer examples <d-cite key="5128907"></d-cite><d-cite key="10.1117/12.2228523"></d-cite>. However, it is often important to perform well on all classes, regardless of their frequency in the dataset. If we train a model to classify a dataset of different skin tumors, most examples may be benign. Still, it is crucial to identify the rare, malignant ones. Experiment design, including training and evaluation methods must therefore be adjusted when using class-imbalanced data. (see Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> section 3.1)
    </p>
    <br/>
</details>
<details>
  <summary><b> Definition: Nuisance Transformation & Transformation Invariance</b></summary>
  <br/>
  <p>
    Transformations are alterations of data. In the context of image classification, <b>nuisance transformations</b> are alterations that do not affect the class labels of the data. A model is said to be invariant to a <b>nuisance transformation</b> if it can successfully ignore the transformation when predicting a class label.
  </p>
    We can formally define a <b>nuisance transformation</b>
  <p>
    $$T(\cdot |x)$$
  </p>
  <p>
    as a distribution over transformation functions. An example of a <b>nuisance transformation</b> might be a distribution over rotation matrices of different angles, or lighting transformations with different exposure values. By definition, <b>nuisance transformations</b> have no impact on class labels $y$, only on data $x$. A perfectly <b>transformation-invariant</b> classifier would thus completely ignore them, i.e.,
  </p>
  <p>
    $$
        \hat{P}_w(y = j|x) = \hat{P}_w(y = j|x'), \; x' \sim T(\cdot |x).
    $$
  </p>
  <p>
  (see Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> section 3.1)
  </p>
</details>

## A closer look at the experiment

Let's take a more detailed look at the experiment Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> conducted:

Zhou et al. take a dataset, e.g., CIFAR-100, then apply a nuisance transformation, for example, random rotation, background intensity, or dilation and erosion. They then remove samples from some classes until the distribution of class sizes follows [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) with parameter 2.0 and a minimum class size of 5. The test set remains balanced, i.e., all test classes have the same number of samples. They then train a CNN model - for example, a ResNet - on this imbalanced and transformed training data. 

To measure the invariance of the trained model to the applied transformation Zhou et al. use the empirical [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback-Leibler_divergence) between the predictions on the untransformed test set and the transformed test set of each class. 

If the learner is invariant to the transformation, the predicted probability distribution over class labels should be identical for the transformed and untransformed images. In that case, the KLD should be zero and greater than zero otherwise. The higher the expected KL-divergence, the more the applied transformation impacts the network's predictions.

The result: eKLD falls with class size. This implies that the CNN does not learn that there are the same nuisance transformations on all images and therefore does not transfer this knowledge to the classes with less training data. A CNN learns invariance **separately for each class**(see also Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> section 3.2). 

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/EKLD.svg" class="img-fluid" %}


## How is this a meta-learning experiment? 

You might think this is a cool experiment, but how is it related to meta-learning? 

And, indeed, in contemporary literature meta-learning is often conceived of as learning multiple tasks. In an much-cited 2022 survey, Hosepdales et al. write:
>Meta-learning is most commonly understood as learning to learn; the process of improving a learning algorithm over multiple learning episodes. In contrast, conventional ML improves model predictions over multiple data instances. <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite>

In another popular survey Vanschoren [2018] says:
>First, we need to collect meta-data that describe prior learning tasks and previously learned models. They comprise the exact algorithm configurations used to train the models, including hyperparameter settings, pipeline compositions and/or network architectures, the resulting model evaluations, such as accuracy and training time, the learned model parameters, such as the trained weights of a neural net, as well as measurable properties of the task itself, also known as meta-features.<d-cite key="vanschoren2018meta"></d-cite>

Francheschi et al. [2018] basically equate meta-learning (ML) with [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization) (HO):
>[...] both HO and ML essentially boil down to nesting two search problems: at the inner level we seek a good hypothesis (as in standard supervised learning) while at the outer level we seek a good configuration (including a good hypothesis space) where the inner search takes place.<d-cite key="DBLP:conf/icml/FranceschiFSGP18"></d-cite> 

This perspective on meta-learning seems to indicate that "true" meta-learning requires a rigid structure of multiple discrete tasks that is optimized over. However, in the invariance transfer setting we neither have multiple learning episodes, i.e., we learn over multiple data instances, nor any "meta-features". Also, adding a class to the dataset does not exactly constitute a new "task", even though knowledge of the nuisance transform is applicable.

So is Zhou et al.'s experiment no meta-learning after all? 

Let's look at one of the original papers on meta-learning. In the 1998 book "Learning to learn" Sebastian Thrun & Lorien Pratt define an algorithm as capable of "Learning to learn" if it improves its performance in proportion to the number of tasks it is exposed to:

>an algorithm is said to learn to learn if its performance at each task improves with experience and with the number of tasks. Put differently, a learning algorithm whose performance does not depend on the number of learning tasks, which hence would not benefit from the presence of other learning tasks, is not said to learn to learn <d-cite key="DBLP:books/sp/98/ThrunP98"></d-cite>

Now this seems a much looser definition. How might this apply to the experiment just outlined? In the introduction, we thought about how a sophisticated learner might handle a dataset like the one described in the last section. We said that a sophisticated learner would learn that the nuisance transformations are applied uniformly **to all classes**. Therefore, if we added more classes to the dataset, the learner would become **more invariant** to the transformations because we expose it to more examples of them. Since this is part of the classification task **for each class**, the learner should, everything else being equal, become better at classification, especially on classes with few training examples. To see this, we must think of the multi-classification task not as a single task but as multiple mappings from image features to activations that must be learned, as a set of binary classification tasks. Thrun and Pratt continue:

>For an algorithm to fit this definition, some kind of *transfer* must occur between multiple tasks that must have a positive impact on expected task-performance <d-cite key="DBLP:books/sp/98/ThrunP98"></d-cite>.

This transfer is what Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> tried to measure. There is some meta-information learnable across several tasks, in our case, the transformation distribution across many binary classification tasks. If a learner can learn this meta-information and transfer it to each new task it has "learned to learn"; it is a meta-learner. The goal of Zhou et al.'s <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> experiment was to see whether this transfer takes place. Thus, arguably, it is a meta-learning experiment. 

## Generative Invariance Transfer

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> don't stop there. They show that using the MUNIT (Multimodal Unsupervised image-to-image Translation)<d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite>  architecture, they can learn the nuisance transformations applied to the dataset and generate additional training samples for the classes with few samples, improving transformation invariance there. They call this Generative invariance transfer (GIT). Let's take a closer look: 

MUNIT networks are capable of performing image-to-image translation, which means that they can translate an image from one domain, such as pictures of leopards, into another domain, such as pictures of house cats. The translated image should look like a real house cat while still resembling the original leopard image. For instance, if the leopard in the original image has its eyes closed, the translated image should contain a house cat with closed eyes. Eye state is a feature present in both domains, so a good translator should not alter it. On the other hand, a leopard's fur is yellow and spotted, while a house cat's fur can be white, black, grey, or brown. To make the translated images indistinguishable from real house cats, the translator must thus replace leopard fur with house cat fur.

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_ENCODING.svg" class="img-fluid" %}

MUNIT networks learn to perform translations by correctly distinguishing the domain-agnostic features (such as eye state) from the domain-specific features (such as the distribution of fur color). They embed an image into two latent spaces: a content space that encodes the domain-agnostic features and a style space that encodes the domain-specific features (see figure above).

To transform a leopard into a house cat, we can encode the leopard into a content and a style code, discard the leopard-specific style code, randomly select a cat-specific style code, and assemble a house cat image that looks similar by combining the leopard's content code with the randomly chosen cat style code (see figure below).

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_TRANSLATION.svg" class="img-fluid" %}

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> modify the process of using MUNIT to transfer images between domains. They do not use MUNIT to translate images **between** domains but **within** a domain. The MUNIT network exchanges the style code of an image with another style code of the same domain. For example, if the domain is house cats, the MUNIT network might translate a grey house cat into a black one. The learning task in this single-domain application of MUNIT is to decompose example-agnostic content features from example-specific style features so that the translated images still look like house cats. For example, fur color is a valid style feature for translating within the 'house cat' domain because every house cat has a fur color. A translator only switching fur color is hard to detect.

 However, if the domain included house cats **and apples**, fur color is not a valid style feature. If it was, the translator might translate fur color on an apple and give it black fur, which would look suspiciously out of place. Whatever house cats and apples have in common - maybe their position or size in the frame - would be a valid style feature. We would expect an intra-domain translator on an apples-and-cats dataset to change the position and size of an apple but not to turn it into a cat (not even partially).

It turns out that on a dataset with uniformly applied nuisance transformations, the nuisance transformations are valid style features: The result of randomly rotating an apple cannot be discerned as artificial when images of all classes, house cats and apples, were previously randomly rotated. 

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> find that when they train a MUNIT network on a dataset with nuisance transformations and class imbalances, the MUNIT network decomposes the class and transformation distributions. The style latent space of the MUNIT network approximates the transformation distribution $T(\cdot &#124;x)$. The content space preserves the remaining features of the image, such as its class. Thus, when translating an image, i.e., exchanging its style code, MUNIT applies a random nuisance transformation while preserving content. Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use this method to generate data for classes with few examples. While the CNN is still unable to transfer invariance to $T(\cdot &#124;x)$ between classes, it can now learn it for each class separately using the data generated by MUNIT, which has acquired knowledge of $T(\cdot &#124;x)$ from the entire dataset (see also Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> section 4).

So MUNIT decomposes the example-specific information, e.g., whether something is an apple or a house cat, from the meta-information, i.e., nuisance transformations applied to the entire dataset. When we add more classes, it has more data and can better learn the transformation distribution $T(\cdot &#124;x)$. Does solving a meta-learning problem make MUNIT a meta-learner? Let's look at the relationship MUNIT has with contemporary meta-learners 

## How much meta-learning is in MUNIT ?
To see how well MUNIT fits the definition of meta-learning, let's see what the same survey papers we consulted earlier consider the structure of a meta-learning algorithm. 

### Part 1: The task-centered view
Hospedales et al. [2021] <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite> defines a generic meta-learner as follows: 
An outer training loop with a set of trainable parameters iterates over tasks in a  distribution of tasks. Formally a task is comprised of a dataset and a loss function $ \mathcal{T} = \\\{ \mathcal{D}, \mathcal{L} \\\} $. In an inner loop, a learning algorithm based on the outer loop's parameters is instantiated for each task. We train it on a training set (*meta-training*) and test it on a validation set (*meta-validation*). We then use loss on this validation set to update the outer loop's parameters. In this task-centered view of meta-learning, we can express the objective function as

<p>
$$
\underset{\omega}{\mathrm{min}} \; \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \; \mathcal{L}(\mathcal{D}, \omega), 
$$
</p>

where $ \omega $ is parameters trained exclusively on the meta-level, i.e., the *meta-knowledge* learnable from the task distribution <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite>.


This *meta-knowledge* is what the meta-learner accumulates and transfers across the tasks. Collecting meta-knowledge allows the meta-learner to improve its expected task performance with the number of tasks. The meta-knowledge in the experiment of Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> is the invariance to the nuisance transformations as the transformations are identical and need to be ignored for images of all classes. By creating additional transformed samples, the MUNIT network makes the meta-knowledge learnable for the CNN.

The task-centered view of meta-learning brings us to a related issue: A meta-learner must discern and decompose task-specific knowledge from meta-knowledge. Contemporary meta-learners decompose meta-knowledge through the different objectives of their inner and outer loops and their respective loss terms. They store meta-knowledge in the outer loop's parameter set $ \omega $ but must not learn task-specific information there. Any unlearned meta-features lead to slower adaptation, negatively impacting performance, *meta-underfitting*. On the other hand, any learned task-specific features will not generalize to unseen tasks in the distribution, thus also negatively impacting performance, *meta-overfitting*.

We recall that, similarly, MUNIT <d-cite key="DBLP:conf/eccv/HuangLBK18"></d-cite> decomposes domain-specific style information and domain-agnostic content information. Applied to two domains, leopards and house cats, a MUNIT network will encode the domain-agnostic information, e.g., posture, scale, background, in its content latent space, and the domain-specific information, e.g., how a cat's hair looks, in its style latent space. If the MUNIT network encoded the domain-agnostic information in the style latent space, the resulting image would not appear to be a good translation since the style information is discarded and replaced. It might turn a closed-eyed leopard into a staring cat. If the MUNIT network encoded the domain-specific transformation in the content latent space, the network would have difficulty translating between domains. A house cat might still have its original leopard fur.

Although the single-domain application of MUNIT explicitly learns a single task and scales "over multiple data instances" instead of "multiple learning episodes"<d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite> it is clearly compatible with the task-centered view of meta-learning set forth *in the same survey paper*. Both meta-learning and multi-domain unsupervised image-to-image translation are thus learning problems that require a separation of the general from the specific. 

As we shall see, this is even visible when comparing their formalizations as optimization problems.

### Part 2: The bi-level programming view

Francheschi et al. [2018] <d-cite key="DBLP:conf/icml/FranceschiFSGP18"></d-cite> show that all contemporary neural-network-based meta-learning approaches can be expressed as bi-level optimization problems. Formally the optimization objective of a general meta-learner can be expressed as:

<p>
$$
\bbox[5pt, border: 2px solid blue]{
\begin{align*}
   \omega^{*} = \underset{\omega}{\mathrm{argmin}} \sum_{i=1}^{M} \mathcal{L}^{meta}(\theta^{* \; (i)}(\omega), D^{val}_i),
\end{align*}
}
$$
</p>


where $M$ describes the number of tasks in a batch, $\mathcal{L}^{meta}$ is the meta-loss function, and $ D^{val}_i $ is the validation set of the task $ i $. $\omega$ represents the parameters exclusively updated in the outer loop. $ \theta^{* \; (i)} $ represents an inner loop learning a task that we can formally express as a sub-objective constraining the primary objective

<p>
$$
\bbox[5pt, border: 2px solid red]{
\begin{align*}
   s.t. \; \theta^{* \; (i)} = \underset{\theta}{\mathrm{argmin}} \; \mathcal{L^{task}}(\theta, \omega, D^{tr}_i),
\end{align*}
}
$$
</p>

where $ \theta $ are the model parameters updated in the inner loop, $ \mathcal{L}^{task} $ is the loss function by which they are updated and $ D^{tr}_i $ is the training set of the task $ i $  <d-cite key="DBLP:journals/pami/HospedalesAMS22"></d-cite>.

While not adhering to Francheschi et al.'s [2018] notion of a meta-learner as "nesting two search problems", it turns out that the loss functions of MUNIT can be similarly decomposed:

{% include figure.html path="assets/img/2022-12-01-how-much-meta-learning-is-in-image-to-image-translation/MUNIT_LOSS.svg" class="img-fluid" %}


MUNIT's loss function consists of two adversarial (GAN) <d-cite key="DBLP:conf/nips/GoodfellowPMXWOCB14"></d-cite> loss terms (see figure above) with several auxiliary reconstruction loss terms. To keep the notation simple, we combine all reconstruction terms into a joined reconstruction loss $ \mathcal{L}_{recon}(\theta_c, \theta_s) $, where $ \theta_c $ are the parameters of the *content* encoding/decoding networks and $ \theta_s $ are the parameters of the *style* encoding/decoding networks. We will only look at one of the two GAN losses in detail since they are symmetric, and one is discarded entirely when MUNIT is used on a single domain in the fashion of Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>.

MUNIT's GAN loss term is

<p>
$$
\begin{align*}
    &\mathcal{L}^{x_{2}}_{GAN}(\theta_d, \theta_c, \theta_s) 
    \\\\
    =& \;\mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right]
    \\
    +& \;\mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right],
\end{align*}
$$
</p>

where the $ \theta_d $ represents the parameters of the discriminator network, $p(x_2)$ is the data of the second domain, $ c_1 $ is the content embedding of an image from the first domain to be translated. $ s_2 $ is a random style code of the second domain. $ D_2 $ is the discriminator of the second domain, and $ G_2 $ is its generator. MUNIT's full objective function is:

<p>
$$
\begin{align*}
        \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \underset{\theta_d}{\mathrm{argmax}}& \;\mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right]
    \\ +& \; \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right], + \; \mathcal{L}^{x_{1}}_{GAN}(\theta_d, \theta_c, \theta_s) 
    \\ +& \;\mathcal{L}_{recon}(\theta_c, \theta_s)
\end{align*}
$$
</p>

(compare <d-cite key="DBLP:conf/eccv/HuangLBK18, DBLP:conf/nips/GoodfellowPMXWOCB14"></d-cite>).
We can reformulate this into a bi-level optimization problem by extracting a minimization problem describing the update of the generative networks.
We also drop the second GAN loss term as it is not relevant to our analysis. 

<p>
$$
\bbox[5px, border: 2px solid blue]{
\begin{align*}
    \omega^{*} 
    & = \{ \theta_c^*, \theta_s^* \} 
    \\\\
    & = 
    \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d^{*})) \right]
    \\
    & + \mathcal{L}_{recon}(\theta_c, \theta_s),
\end{align*}
}
$$
</p>

We then add a single constraint, a subsidiary maximization problem for the discriminator function:

<p>
$$
\bbox[5px, border: 2px solid red]{
\begin{align*}
   &s.t. \;\theta_d^{*}
   \\\\
    & =
    \underset{\theta_d}{\mathrm{argmax}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right] 
    \\
    & + \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right]
\end{align*}
}
$$
</p>

Interestingly, this bi-level view does not only resemble a meta-learning procedure as expressed above, but the bi-level optimization also facilitates a similar effect. Maximizing the discriminator's performance in the constraint punishes style information encoded as content information. If style information is encoded as content information, the discriminator detects artifacts of the original domain in the translated image. Similarly, a meta-learner prevents *meta-overfitting* via an outer optimization loop. 

*However, MUNIT, while representable as a bi-level optimization problem does not "essentially boil down to nesting two search problems".<d-cite key="DBLP:conf/icml/FranceschiFSGP18"></d-cite>* During GAN training, the discriminator's parameters are updated through the changes in the generator's parameters, which derive from the discriminator's parameters, and so forth; The training of the discriminator and generator are dependent processes. Crucially, they depend on each other symmetrically, forming a min-max game. Contemporary meta-learners, meanwhile, are strictly hierarchical, with an outer and inner optimization loop.

### Now, does MUNIT meta-learn?

So it appears that while not conforming to any verbal definition of a contemporary meta-learner MUNIT seems to:

a) adhere to multiple formalizations made in the very same publications to define meta-learning

b) solve a meta-learning problem via GIT when applied to a single domain (if you agree with the conclusion of the previous chapter)

We thus conclude:

When applied to a single domain MUNIT *does* meta-learn as it combines information from all classes to extract the transformation distribution. While it does not perform classification explicitly, the class information of an image is encoded in MUNIT's content space. Since MUNIT is trained in an unsupervised way, it is probably closer to a distance metric than an actual class label. We might thus classify single-domain MUNIT as an unsupervised, generative meta-learner. 

## Implications

That invariance transfer and GIT are meta-learning and that MUNIT is a meta-learner is important. Granted, it is not especially hard to see that invariance transfer is a form of "learning to learn" or that Image-to-Image translation is essentially a mechanism to decompose class-specific form general features. 

However, because contemporary meta-learning has been narrowly cast as "improving a learning algorithm over multiple learning episodes"<d-cite key="DBLP:journals/pami/HospedalesAMS22"> and "nesting two search problems"<d-cite key="DBLP:conf/icml/FranceschiFSGP18"> it is hard to recognize GIT as meta-learning. 

In these authors opinion this is not GIT's fault, but a sign that meta-learning has recently been conceived of too narrowly. Zhou et al.'s [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> experiment is a beautiful illustration of this showing that something as general as a GAN loss term, with appropriate modifications, can be used to meta-learn. 

A too-narrow conception goes further than obscuring some experiment's significance though: Meta-learning as a field has recently struggled to compete with less specialized architectures<d-cite key="DBLP:journals/corr/abs-2104-02638"></d-cite><d-cite key="DBLP:journals/corr/abs-1904-04232"></d-cite>. Multi-task datasets are hard to scale <d-cite key="triantafillou2019meta"></d-cite>, as are episode rollouts <d-cite key="DBLP:conf/icml/FinnAL17"></d-cite>. Meanwhile, large-scale architectures have shown impressive zero-shot capabilities<d-cite key="dosovitskiy2021an"></d-cite><d-cite key="pmlr-v139-radford21a"></d-cite>.

Zhou et al.'s [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> contributions are therefore important as a challenge to the status quo in meta-learning. MUNIT seems to meta-learn by embedding class (and class-specific features) in one space and transformation-specific features (e.g., how bright/dark) in another. This seems to point to a conception of meta-learning as finding mutual information between sets of examples (not necessarily defined by class or transformation feature but by arbitrary concepts) or hierarchies of such sets. Examining and designing mechanisms by which such behavior can be evoked is an exciting direction for future work.

## Key Takeaways

1. Zhou et al.'s <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> experiments show that the meta-learning setting can be formulated more broadly than learning an explicit task distribution, suggesting that specialized datasets are not necessary.

2. Using GIT, Zhou et al. <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> show that meta-learning algorithms can come in shapes other than inner and outer training loops. Analysis suggests that countervailing loss terms facilitate the decomposition of meta-features from task-specific features.

3. Our discussion of Zhou et al.'s <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> experiments suggests, that when thinking about meta-learning, thinking about mutual information between batches of examples (not necessarily aligned with class labels) and how to extract it trumps thinking about distinct tasks.
