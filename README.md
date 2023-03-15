## Transformer Encoder with Multiscale Deep Learning for Pain Classification Using Physiological Signals


### Our new paper just released on arxiv, PainAttnNet: https://arxiv.org/abs/2303.06845 

<br>

### Code will be coming soon.

## Abstract

<p align="center">
  <img src="figures/outline_simple_2.svg" width="100%"/>
</p>

Pain is a serious worldwide health problem that affects a vast proportion of the population. For efficient pain management and treatment, accurate classification and evaluation of pain severity are necessary. However, this can be challenging as pain is a subjective sensation-driven experience. Traditional techniques for measuring pain intensity, e.g. self-report scales, are susceptible to bias and unreliable in some instances. Consequently, there is a need for more objective and automatic pain intensity assessment strategies. In this research, we develop PainAttnNet (PAN), a novel transformer-encoder deep-learning framework for classifying pain intensities with physiological signals as input. The proposed approach is comprised of three feature extraction architectures: multiscale convolutional networks (MSCN), a squeeze-and-excitation residual network (SEResNet), and a transformer encoder block. On the basis of pain stimuli, MSCN extracts short- and long-window information as well as sequential features. SEResNet highlights relevant extracted features by mapping the interdependencies among features. The third architecture employs a transformer encoder consisting of three temporal convolutional networks (TCN) with three multi-head attention (MHA) layers to extract temporal dependencies from the features. Using the publicly available BioVid pain dataset, we test the proposed PainAttnNet model and demonstrate that our outcomes outperform state-of-the-art models. These results confirm that our approach can be utilized for automated classification of pain intensity using physiological signals to improve pain management and treatment.


## Citation

arxiv
```
@misc{lu2023transformer,
      title={Transformer Encoder with Multiscale Deep Learning for Pain Classification Using Physiological Signals}, 
      author={Zhenyuan Lu, Burcu Ozek and Sagar Kamarthi},
      year={2023},
      eprint={2303.06845},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
