# ProPhyT
**Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window Segmentation Under Missing Perfusion**

---

## Abstract

Accurate delineation of ischemic core and salvageable penumbra is critical for guiding reperfusion therapy in acute ischemic stroke (AIS). While CT perfusion (CTP) provides quantitative hemodynamic parameters for tissue-window assessment, its availability remains limited in many emergency settings. In contrast, non-contrast CT (NCCT) is nearly universally accessible but lacks explicit perfusion information, making core and penumbra segmentation highly challenging.

We study stroke tissue-window segmentation under **missing perfusion**, where only NCCT is available at inference. In this setting, specialist NCCT models struggle due to subtle contrast and limited supervision, while reconstructing CTP from NCCT constitutes an ill-posed inverse problem prone to hemodynamic hallucination.

To address this, we propose **ProPhyT**, a prototype-based physiological transfer framework that injects CTP-derived hemodynamic semantics into NCCT-based segmentation **without reconstructing perfusion maps**.

---

## Method

ProPhyT consists of three key processes:

1. **Hemodynamic Prototype Support Bank Construction**
   A self-supervised CTP encoder learns a hemodynamic embedding space. CTP features are then clustered to form representative physiological prototypes that capture tissue-level perfusion states.

2. **Cross-Modal Prototype Retrieval Learning**
   An NCCT encoder is aligned to the CTP embedding space, enabling the retrieval of hemodynamic prototypes directly from NCCT features at inference time — without requiring CTP input.

3. **Physiological Prototype-Conditioned Segmentation**
   Retrieved prototypes are converted into similarity-aware location prompts to parameter-efficiently fine-tune a medical foundation segmentation model for ischemic core and penumbra prediction.

![ProPhyT Framework Overview](assets/framework.png)

---

## Results

Extensive experiments demonstrate consistent improvements over top-performing baselines for ischemic core and penumbra segmentation, supporting more reliable reperfusion decision-making in CTP-limited clinical settings.

---

## Code

Code will be available upon acceptance.

---

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{prophyt2025,
  title   = {Prototype-based Physiological Transfer Enables NCCT-only Stroke Tissue-Window Segmentation Under Missing Perfusion},
  author  = {},
  journal = {},
  year    = {2025}
}
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
