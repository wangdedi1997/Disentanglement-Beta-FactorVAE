# Disentanglement-Beta-FactorVAE

This work is done by Dedi Wang during the first rotation (2019.09-2019.11) in Prof. Pratyush Tiwary's lab. The goal of this project is to test the performance of VAE-based disentanglement algorithms on two toy models.

## Methods

### Toy Model
In my first model, x1 and x2 are produced from two independent Gaussian distribution, where the mean is zero and the variance is 1. Then I entangled them to produce a 2-dimensional entangled dataset. Of course, we also add a noise to simulate the real situation.   

<img src="https://render.githubusercontent.com/render/math?math=x_1'=x_1+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_2'=(x_1+x_2)/\sqrt{2}+\epsilon">

And my second toy model is similar, but instead of outputting a 2-dimensional entangled dataset, now we output an 8-dimensional entangled dataset. 

<img src="https://render.githubusercontent.com/render/math?math=x_1'=(x_1+2x_2)/\sqrt{5}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_2'=(x_1-2x_2)/\sqrt{5}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_3'=(2x_1+x_2)/\sqrt{5}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_4'=(2x_1-2x_2)/\sqrt{5}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_5'=(x_1+x_2)/\sqrt{2}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_6'=-(x_1+x_2)/\sqrt{2}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_7'=(x_1-x_2)/\sqrt{2}+\epsilon">
<img src="https://render.githubusercontent.com/render/math?math=x_8'=-(x_1-x_2)/\sqrt{2}+\epsilon">

### Algorithm
In this project, I studied the performance of FactorVAE, proposed in the paper Disentangling by Factorising (Kim & Mnih, 2018) [https://arxiv.org/pdf/1802.05983.pdf](https://arxiv.org/pdf/1802.05983.pdf), on the two toy models. A small revision is made in the object function: a hyper-parameter $\beta$ is added to change the bottleneck capacity. Thus, our object fuction is 

<img src="http://latex.codecogs.com/gif.latex?L%3D-%5Cfrac%7B1%7D%7BN%7D%5Csum%5Climits_%7Bi%3D1%7D%5E%7BN%7D%5Cleft%5BE_%7Bq%28z%7Cx%5Ei%29%7D%5B%5Clog%20p%28x%5Ei%7Cz%29%5D&plus;%5Cbeta%20KL%28q%28z%7Cx%5Ei%29%7C%7Cp%28z%29%29%5Cright%5D&plus;%5Cgamma%20KL%28q%28z%29%7C%7C%5Cbar%7Bq%7D%28z%29%29">

Here, the total correlation of latent variables is also estimated using the density ratio trick as FactorVAE did. 
