# Medical VLMs OOD Detection
Delving into Out-of-Distribution Detection with Medical Vision-Language Models.

Recent advances in medical vision-language models (VLMs) demonstrate impressive performance in image classification tasks, driven by their strong zero-shot generalization capabilities. However, given the high variability and complexity inherent in medical imaging data, the ability of these models to detect out-of-distribution (OOD) data in this domain remains underexplored. In this work, we conduct the first systematic investigation into the OOD detection potential of medical VLMs. We evaluate state-of-the-art VLM-based OOD detection methods across a diverse set of medical VLMs, including both general and domain-specific purposes. To accurately reflect real-world challenges, we introduce a cross-modality evaluation pipeline for benchmarking full-spectrum OOD detection, rigorously assessing model robustness against both semantic shifts and covariate shifts. Furthermore, we propose a novel hierarchical prompt-based method that significantly enhances OOD detection performance. Extensive experiments are conducted to validate the effectiveness of our approach.



## Introduction
This repository provides some main examples for our work to help you understand the evaluation of OOD detection using vision-language models.

## Installation
There is no unified environment for this work since they are highly built on the existing VLMs works.

To use our provided jupyter notebooks, you should build the environment based on the following repositories:

| Domain        | VLM        | Github link                                |
|---------------|------------|--------------------------------------------|
| Ophthalmology | FLAIR      | https://github.com/jusiro/FLAIR            |
| Radiology     | UniMedCLIP | https://github.com/mbzuai-oryx/UniMed-CLIP |
| Pathology     | QuilNet    | https://github.com/wisdomikezogwo/quilt1m  |

Notes: FLAIR uses a default template "A fundus photograph of [CLS]", please modify ```flair/modeling/model.py#L28``` to caption="" for the naive prompt setting without providing image modality information.
```python
class FLAIRModel(torch.nn.Module):
    def __init__(self, vision_type='resnet_v1', bert_type='emilyalsentzer/Bio_ClinicalBERT', vision_pretrained=True,
                 proj_dim=512, proj_bias=False, logit_scale_init_value=0.07, from_checkpoint=True, weights_path=None,
                 out_path=None, image_size=512, caption="A fundus photograph of [CLS]", projection=True,
                 norm_features=True):
```
## Prepare Datasets
Fives: https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1

LC25000: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

X-ray: https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset?select=normal

CT: https://www.kaggle.com/datasets/anaselmasry/covid19normalpneumonia-ct-images

ImageNet: https://drive.google.com/drive/folders/1wIl7Pk0SXmZY5fhyWTtd4W6VQYP1FAg6?usp=sharing

Extract the datasets and place them as jupyter notebook files indicate.

## Citation
If you find this work useful, we kindly request you to cite our paper:

```
@article{ju2025delving,
  title={Delving into Out-of-Distribution Detection with Medical Vision-Language Models},
  author={Ju, Lie and Zhou, Sijin and Zhou, Yukun and Lu, Huimin and Zhu, Zhuoting and Keane, Pearse A and Ge, Zongyuan},
  journal={arXiv preprint arXiv:2503.01020},
  year={2025}
}
```
