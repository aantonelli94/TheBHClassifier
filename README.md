# TheBHClassifier

A repository of codes to classify the generation and formation channels of dynamically-formed gravitational-wave events using machine-learning algorithms.


This project is useful to all those who want to quickly inspect the main metrics of the dynamical interactions of an observed black-hole binary -- such as the generation and formation channel -- and it could also be useful as a benchmark analysis for further investigation of the hyperparameters of the globular clusters.

The analysis relies on [rapster](https://github.com/Kkritos/Rapster) simulations, and it assumes that these simulations represent fairly well the environs of globular clusters. The second main implicit assumption is that the binaries used in the event-based analysis are formed dynamically.

*The repo is under active development.*


## Get started

The codes folder contains the following files:

1. ` model_selection.ipynb`, where the model selection is performed and a random forest (RF) classifier is chosen.
2. ` model_training.py`, where the RF classifier is trained on simulations from [rapster](https://github.com/Kkritos/Rapster) and the output is stored in ` models`.
3. ` event_based_analysis.py`, where data from LIGO is input in the trained models to get likelihood of formation path and generation.
4. A ` results` folder where the output of the previous step can be stored.
5. ` plot_probabilities.py`, where the results can be plotted.

While the repo is as self-contained as possible, external data must be downloaded. 

` model_training.py` requires simulations from rapster, which can be found [at this link](https://zenodo.org/record/7358638#.Y3_cPuzMK3I). The model has been validated on 7 simulations with different assumptions on the half-mass radius, initial slope of the cluster and initial distribution of the black-hole spins. We recommend downloading the simulations into a `data` folder in the codes' folder.

` event_based_analysis.py` requires downloading gravitational-wave data from [the second](https://zenodo.org/record/6513631#.Y4De3-zMK3J) and [third catalog](https://zenodo.org/record/5546663#.Y4DepezMK3J) into the `data` folder. Examples of binaries that are suited for this investigation are GW190521, GW190412, GW191109 and GW200225, for which the following data could be downloaded:

- ` IGWN-GWTC2p1-v2-GW190412_053044_PEDataRelease_mixed_cosmo.h5`
- ` IGWN-GWTC2p1-v2-GW190521_030229_PEDataRelease_mixed_cosmo.h5`
- ` IGWN-GWTC3p0-v1-GW191109_010717_PEDataRelease_mixed_cosmo.h5`
- ` IGWN-GWTC3p0-v1-GW200225_060421_PEDataRelease_mixed_cosmo.h5`


## Usage


To plot the probabilities type the following in terminal

` python plot_probabilities.py --event 'GW190521' `

Changing the name of the event as needed. Notice that the argument must be a string, and that the code will run only if ` event_based_analysis.py` has been run on the input event.


## Requirements

```
matplotlib==3.5.2
numpy==1.21.5
pandas==1.4.3
scikit_learn==1.1.3
seaborn==0.11.2
```

## Report a bug

You can send me an email at aantone3@jh.edu. All feedback is appreciated!
