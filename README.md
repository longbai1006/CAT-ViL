

<div align="center">

<samp>

<h2> Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery </h1>

<h4> Long Bai*, Mobarakol Islam*, and Hongliang Ren </h3>

</samp>   

</div>     

---

If you find our code or paper useful, please cite as

```bibtex
@article{bai2023co,
  title={Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery},
  author={Bai, Long and Islam, Mobarakol and Ren, Hongliang},
  journal={arXiv preprint arXiv:2307.05182},
  year={2023}
}
```

---
## Abstract

Medical students and junior surgeons often rely on senior surgeons and specialists to answer their questions when learning surgery. However, experts are often busy with clinical and academic work, and have little time to give guidance. Meanwhile, existing deep learning (DL)-based surgical Visual Question Answering (VQA) systems can only provide simple answers without the location of the answers. In addition, vision-language (ViL) embedding is still a less explored research in these kinds of tasks. We develop a surgical Visual Question Localized-Answering (VQLA) system to help medical students and junior surgeons learn and understand from recorded surgical videos. We propose an end-to-end Transformer with Co-Attention gaTed Vision-Language (CAT-ViL) embedding for VQLA in surgical scenarios, which does not require feature extraction through detection models. The CAT-ViL embedding module is carefully designed to fuse heterogeneous features from visual and textual sources. The fused embedding will feed a standard Data-Efficient Image Transformer (DeiT) module, before the parallel classifier and detector for joint prediction. We conduct the experimental validation on public surgical videos from MICCAI EndoVis Challenge 2017 and 2018. The experimental results highlight the superior performance and robustness of our proposed model compared to the state-of-the-art approaches. Ablation studies further prove the outstanding performance of all the proposed components. The proposed method provides a promising solution for surgical scene understanding, and opens up a primary step in the Artificial Intelligence (AI)-based VQLA system for surgical training.


---
## Environment

- PyTorch
- numpy
- pandas
- scipy
- scikit-learn
- timm
- transformers
- h5py

## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `EndoVis-18-VQLA/` : Each sequence folder follows the same folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from EndoVIS18 challange.
            - `vqla`
                - `label`: Q&A pairs and bounding box label.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                    - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
        - `....`
        - `seq_16`
    - `EndoVis-17-VQLA/` : 97 frames are selected from EndoVIS17 challange for external validation. 
        - `left_frames`
        - `vqla`
            - `label`: Q&A pairs and bounding box label.
            - `img_features`: Contains img_features extracted from each frame with different patch size.
                - `5x5`: img_features extracted with a patch size of 5x5 by ResNet18.
                - `frcnn`: img_features extracted by Fast-RCNN and ResNet101.
    - `featre_extraction/`:
        - `feature_extraction_EndoVis18-VQLA-frcnn.py`: Used to extract features with Fast-RCNN and ResNet101.
        - `feature_extraction_EndoVis18-VQLA-resnet`: Used to extract features with ResNet18 (based on patch size).
- `models/`: 
    - GatedLanguageVisualEmbedding.py : GLVE module for visual and word embeddings and fusion.
    - LViTPrediction.py : our proposed LViT model for VQLA task.
    - VisualBertResMLP.py : VisualBERT ResMLP encoder from Surgical-VQA.
    - visualBertPrediction.py : VisualBert encoder-based model for VQLA task.
    - VisualBertResMLPPrediction.py : VisualBert ResMLP encoder-based model for VQLA task.
- dataloader.py
- train.py
- utils.py

---
## Dataset (will release after acceptance)
1. EndoVis-18-VQLA
    - [Images](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/) 
    - VQLA
2. EndoVis-17-VQLA
    - [Images](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/Home/)
    - VQLA  

---

## Run training
- Train on EndoVis-18-VLQA 
    ```bash
    python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 64 --epochs 80
    ```

---
## Evaluation
- Evaluate on both EndoVis-18-VLQA & EndoVis-17-VLQA
    ```bash
    python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 64
    ```
