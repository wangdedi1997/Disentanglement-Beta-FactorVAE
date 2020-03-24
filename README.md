# Disentanglement-Beta-FactorVAE


This work is done by Dedi Wang during the first rotation (2019.09-2019.11) in Prof. Pratyush Tiwary's lab. The goal of this project is to test the performance of VAE-based disentanglement algorithms on two toy models.

## Methods

### Toy Model
In my first model, $x_1$ and $x_2$ are produced from two independent Gaussian distribution, where the mean is zero and the variance is 1. Then I entangled them to produce a 2-dimensional entangled dataset. Of course, we also add a noise to simulate the real situation. 
$x_1'=x_1+\epsilon$
$x_2'=(x_1+x_2)/\sqrt{2}+\epsilon$
And my second toy model is similar, but instead of outputting a 2-dimensional entangled dataset, now we output an 8-dimensional entangled dataset. 
$x_1'=(x_1+2x_2)/\sqrt{5}+\epsilon$
$x_2'=(x_1-2x_2)/\sqrt{5}+\epsilon$
$x_3'=(2x_1+x_2)/\sqrt{5}+\epsilon$
$x_4'=(2x_1-2x_2)/\sqrt{5}+\epsilon$
$x_5'=(x_1+x_2)/\sqrt{2}+\epsilon$
$x_6'=-(x_1+x_2)/\sqrt{2}+\epsilon$
$x_7'=(x_1-x_2)/\sqrt{2}+\epsilon$
$x_1'=-(x_1-x_2)/\sqrt{2}+\epsilon$

### Algorithm
Based on the paper Disentangling by Factorising (Kim & Mnih, 2018) [https://arxiv.org/pdf/1802.05983.pdf](https://arxiv.org/pdf/1802.05983.pdf), our target of object function is 
$L=-\frac{1}{N}\sum\limits_{i=1}^{N}\left[E_{q(z|x^i)}[logp(x^i|z)]+\beta KL(q(z|x^i)||p(z))\right]+\gamma KL(q(z)||\bar{q}(z))$

