# Glaucoma-Identification
Optical Disc/Cup segmentation of retinal images using my implementation of UNet++. 

<p align="center">
 <b>This Repository is still a WORK IN PROGRESS</b>
</p>

## Model: UNet++ 
<img src="https://miro.medium.com/max/658/1*ExIkm6cImpPgpetFW1kwyQ.png" width="450" height="250">
UNet++ adds on to the original UNet by adding: redesigned skip pathways, dense skip connections, and deep supervision. \
In this repository I have changed the UNet++ architecture a bit by adding residual connections and dropout layers 

## Dataset: REFUGE (from Grand-Challenge.org)
Contains Retinal Fundus images along with its corresponding mask segmenting the optic nerve head.
