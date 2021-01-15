# Glaucoma-Identification
Optical Disc/Cup segmentation of retinal images using my implementation of UNet++. 

## Model: UNet++ 
<img src="https://miro.medium.com/max/658/1*ExIkm6cImpPgpetFW1kwyQ.png" width="320" height="200">
<a href="https://arxiv.org/abs/1807.10165">UNet++ Paper</a>

<hr>

<p>
 UNet++ adds on to the original UNet by adding: redesigned skip pathways, dense skip connections, and deep supervision. 
In this repository I have changed the UNet++ architecture a bit by adding residual connections and dropout layers
</p>

## Dataset: REFUGE (from Grand-Challenge.org)
Contains Retinal Fundus images along with its corresponding mask segmenting the optic nerve head.
