# OpenBias: Open-set Bias Detection in Text-to-Image Generative Models
[[`arXiv`](https://arxiv.org/abs/2404.07990)][[`pdf`](https://arxiv.org/pdf/2404.07990.pdf)][[`BibTeX`](#BibTeX)]

[Moreno D`Incà](https://scholar.google.com/citations?user=tdTJsOMAAAAJ&hl), [Elia Peruzzo](https://helia95.github.io/), [Massimiliano Mancini](https://mancinimassimiliano.github.io/), [Dejia Xu](https://ir1d.github.io/), [Vidit Goel](https://vidit98.github.io/), [Xingqian Xu](https://xingqian2018.github.io/), [Zhangyang Wang](https://vita-group.github.io/), [Humphrey Shi](https://www.humphreyshi.com/home), [Nicu Sebe](https://disi.unitn.it/~sebe/)

>**Abstract:** Text-to-image generative models are becoming increasingly popular and accessible to the general public. As these models see large-scale deployments, it is necessary to deeply investigate their safety and fairness to not disseminate and perpetuate any kind of biases. However, existing works focus on detecting closed sets of biases defined a priori, limiting the studies to well-known concepts. In this paper, we tackle the challenge of open-set bias detection in text-to-image generative models presenting OpenBias, a new pipeline that identifies and quantifies the severity of biases agnostically, without access to any precompiled set. OpenBias has three stages. In the first phase, we leverage a Large Language Model (LLM) to propose biases given a set of captions. Secondly, the target generative model produces images using the same set of captions. Lastly, a Vision Question Answering model recognizes the presence and extent of the previously proposed biases. We study the behavior of Stable Diffusion 1.5, 2, and XL emphasizing new biases, never investigated before. Via quantitative experiments, we demonstrate that OpenBias agrees with current closed-set bias detection methods and human judgement.

## Code
The code will be available soon! Stay tuned!

## BibTeX
```
@misc{dincà2024openbias,
      title={OpenBias: Open-set Bias Detection in Text-to-Image Generative Models}, 
      author={Moreno D'Incà and Elia Peruzzo and Massimiliano Mancini and Dejia Xu and Vidit Goel and Xingqian Xu and Zhangyang Wang and Humphrey Shi and Nicu Sebe},
      year={2024},
      eprint={2404.07990},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
