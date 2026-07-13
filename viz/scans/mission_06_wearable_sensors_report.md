# Mission 06: Wearable Sensor Processing (2020-2026)

**Mission ID:** 06  
**Topic:** Wearable Sensor Processing  
**Period:** 2020-2026  
**Papers surveyed:** 27  
**Date:** 2026-07-13  

---

## Executive Summary

This scan covers 27 papers on wearable sensor data processing methods from 2020-2026, organized into seven thematic areas: foundation models, imputation, signal quality assessment, missing data handling, sensor fusion for activity recognition, bias/fairness, and negative results. The field has undergone a paradigm shift toward sensor foundation models (NormWear, LSM, Inertia-1, StepFM) that learn generalizable representations from large-scale unlabeled wearable data, while classical imputation and signal-quality methods remain essential for handling the pervasive missingness and noise in real-world deployments.

---

## 1. Wearable Foundation Models

The most significant development in the 2020-2026 period is the emergence of sensor foundation models. These models are pretrained on massive corpora of wearable sensor data and fine-tuned for diverse downstream tasks.

### NormWear and Successors

**NormWear** (Luo et al., 2024) is the first multimodal wearable foundation model, designed to extract generalizable 768-dim representations from heterogeneous sensor configurations (PPG, ECG, EEG, GSR, IMU). It uses a channel-aware attention mechanism with a shared [CLS] token to capture both intra-sensor and inter-sensor patterns. NormWear was pretrained on multiple public datasets and evaluated across 11 datasets spanning 18 applications including mental health, vital sign estimation, and disease risk evaluation. It consistently outperforms baselines under zero-shot, partial-shot, and full-shot settings.

**NormWear-2** (Luo et al., 2026) extends the framework into a world model for physiological signals, incorporating chaos-theoretic balancing of dynamical regime diversity and latent state transition adaptation. It forecasts across multiple temporal scales conditioned on clinical interventions, handling datasets from daily life fitness planning to surgical monitoring. **Key insight:** A smaller but chaos-balanced training corpus can outperform one twice its size by capturing bifurcation regimes in physiological dynamics.

### LSM (Large-Scale Multimodal Model)

**LSM** (Narayanswamy et al., 2024) was trained on 40 million hours of in-situ sensor data from 165,000 people, making it the largest wearable foundation model to date. It covers heart rate, HRV, EDA, accelerometer, skin temperature, and altimeter data. LSM establishes scaling laws for wearable sensor models across compute, data, and model size, with imputation, interpolation, and extrapolation as core tasks.

### Other Foundation Models

- **StepFM** (Wu et al., 2026): A privacy-preserving foundation model built solely on step counter data, transferred to >20 health risk prediction tasks.
- **Inertia-1** (Xu et al., 2026): Open exploration of motion foundation models using >18.2M hours of accelerometer data across 15 datasets.
- **PPG Foundation Model** (Geenjaar et al., 2026): Uses ECG/respiratory signals for multimodal supervision during pretraining; improves 14/15 tasks with 3x fewer subjects.
- **BCG-FM** (Kjaer et al., 2026): First foundation model for ballistocardiography; 3.26-year MAE on biological age estimation from 2.75M hours of nightly recordings.
- **SiamQuality** (Ding et al., 2024): ConvNet-based foundation model robust to imperfect PPG data; outperforms transformer backbones on noisy physiological signals.

---

## 2. Imputation for Wearable Sensor Data

Missing data is endemic in wearable sensing due to non-wear, sensor disconnection, and signal dropout. Several papers address imputation at different granularities.

### Epoch-Level Imputation for Trials

**Tackney et al. (2021, 2023)** developed a comprehensive framework for handling missing accelerometer data in clinical trials. Their approach defines missingness at the epoch level (5-second intervals) rather than the day level, preserving temporal information. They propose both parametric and non-parametric multiple imputation (MI) approaches, where non-parametric donor-based MI replaces missing periods with data from the same person or a matched participant. Validated on the PACE-UP Trial, this approach reduces bias in treatment effect estimates compared to standard day-level thresholding.

### Deep Imputation Methods

- **DynImp** (Huo et al., 2022): Combines nearest-neighbor imputation along the feature axis with an LSTM-based denoising autoencoder for temporal reconstruction. Tested at >50% missing rates on activity recognition, demonstrating that multisensory correlations can reconstruct data under extreme missingness.
- **VCR** (Weng et al., 2026): A self-supervised framework using an orthogonal tokenizer and missing-aware Mixture-of-Experts. Reconstructs only shared components across modalities to avoid hallucination. Evaluated under single- and multiple-modality missingness.

---

## 3. Signal Quality Assessment

Given the pervasive noise in wearable PPG and accelerometer data, signal quality assessment is a critical preprocessing step.

### Unsupervised and Lightweight SQA

- **Self-Supervised + Topological SQA** (Shao et al., 2025): First fully unsupervised PPG SQA pipeline. Contrastive 1D-ResNet-18 on 276h unlabeled data + topological signatures via persistent homology. Cross-device without re-tuning (Silhouette 0.72).
- **Lightweight ResNet SQA** (Zhao et al., 2025): Squeeze-and-Excitation ResNet with >99% parameter reduction; 96.52% AUC for PPG quality classification on clinical AF data.

### Motion Artifact Removal

- **Convolutional Sparse Coding** (Basso et al., 2025): Algorithm unfolding for interpretable PPG denoising. SNR from -7.07 to 11.23 dB; 55% HR MAE reduction (synthetic), 23% on PPG-DaLiA.
- **SpO2-Guided Reconstruction** (Liang et al., 2026): Stage-wise time-frequency reconstruction of low-quality dual-wavelength PPG guided by pretrained SpO2 predictor. MAE 2.882%.
- **UniCardio** (Chen et al., 2025): Unified diffusion transformer for PPG/ECG/BP denoising, imputation, and translation. Exploits complementary cardiovascular signal information.

---

## 4. Missing Data and Incomplete Modalities

- **MuteBench** (Zheng et al., 2026): Evaluates 6 fusion architectures under modality missing and within-modality missing across 9 datasets (125K samples). Architecture family is the strongest robustness predictor. Channel-independent models tolerate missing well but are sensitive to within-modality gaps.
- **DEM** (Singh et al., 2026): Glass-box anomaly detection distilling XGBoost into decision trees. AUC 0.9964; 1235x faster than SHAP. Handles sensor faults and missing data.
- **MCSTN** (Fan et al., 2026): Dual-level corruption modeling with manifold-consistent learning for robust HAR under missing measurements and sensor failures.

---

## 5. Sensor Fusion for Activity Recognition

- **DecomposeWHAR** (Xie et al., 2025): Decomposes multi-sensor signals into intra- and inter-sensor components using depthwise separable conv, SSM, and self-attention. Outperforms shared-kernel approaches.
- **Fusion Comparison** (Mohamady et al., 2026): Head-to-head comparison of 7 techniques on HARMES. Gated multimodal fusion achieves macro F1=0.82, +6pp over concatenation baseline.

---

## 6. Bias, Fairness, and Calibration

- **FairTune** (Panchumarthi et al., 2025): Fine-tuning PPG foundation models widens gender fairness gaps despite 80% MAE reduction. GroupDRO and inverse frequency weighting mitigate without accuracy loss.
- **FOG Detection Bias** (Odonga et al., 2025): Bias across age, sex, disease duration, and FOG phenotype in wearable HAR. Conventional mitigation fails; transfer learning improves fairness.
- **Bayesian AFT** (Yang et al., 2026): Measurement error correction for wearable accelerometer data. Subgroup effects vary by race/region for stroke risk.
- **WeBe Sleep Tracking** (Shao et al., 2026): Cross-device calibration for accelerometer sleep/wake classification. Global threshold generalizes across devices (mean TST error 27.4 min).

---

## 7. Negative and Null Results

- **BCoughBench** (Sanap et al., 2026): 5 respiratory FMs degrade substantially under body-coupled wearable conditions. Mean AUROC 0.785 to 0.689-0.723. Sex classification collapses (0.954 to 0.596).
- **Electrostatic HAR** (Bian et al., 2025): Novel electrostatic field sensing underperforms vs accelerometer for HAR. Useful only as complementary modality (+16% for collaborative tasks).

---

## 8. Implications for Config-Driven RL for JITAIs

1. **NormWear validated**: 768-dim embedding from PPG/accelerometer established and generalizable.
2. **FM scaling works**: LSM, Inertia-1 confirm scaling laws and benefits for imputation.
3. **Signal quality essential**: SQA methods needed; unsupervised topological SQA promising for cross-device.
4. **Missing data maturing**: From epoch-level MI to deep imputation, validated approaches exist.
5. **Bias risk real**: FairTune shows FM fine-tuning amplifies disparities; bias-aware adaptation needed.
6. **Negative results matter**: BCoughBench shows FM degradation under realistic conditions.

---

## 9. Selected References

| # | Paper | Year | Category |
|---|-------|------|----------|
| 1 | Luo et al., NormWear | 2024 | Foundation Model |
| 2 | Luo et al., NormWear-2 | 2026 | Foundation Model |
| 3 | Narayanswamy et al., LSM | 2024 | Foundation Model |
| 4 | Wu et al., StepFM | 2026 | Foundation Model |
| 5 | Xu et al., Inertia-1 | 2026 | Foundation Model |
| 6 | Ding et al., SiamQuality | 2024 | Signal Quality |
| 7 | Tackney et al., MI for accelerometer trials | 2023 | Imputation |
| 8 | Huo et al., DynImp | 2022 | Imputation |
| 9 | Weng et al., VCR | 2026 | Missing Data |
| 10 | Shao et al., Self-Supervised PPG SQA | 2025 | Signal Quality |
| 11 | Zhao et al., Lightweight PPG SQA | 2025 | Signal Quality |
| 12 | Basso et al., PPG Sparse Coding Denoising | 2025 | Signal Quality |
| 13 | Chen et al., UniCardio | 2025 | Signal Quality |
| 14 | Panchumarthi et al., FairTune | 2025 | Bias/Fairness |
| 15 | Odonga et al., FOG Detection Bias | 2025 | Bias/Fairness |
| 16 | Sanap et al., BCoughBench | 2026 | Negative Result |
| 17 | Bian et al., Electrostatic HAR | 2025 | Negative Result |
| 18 | Fan et al., MCSTN | 2026 | Missing Data |
| 19 | Zheng et al., MuteBench | 2026 | Missing Data |
| 20 | Xie et al., DecomposeWHAR | 2025 | Sensor Fusion |
| 21 | Mohamady et al., Fusion Comparison | 2026 | Sensor Fusion |
| 22 | Yang et al., Bayesian AFT Measurement Error | 2026 | Calibration |
| 23 | Shao et al., WeBe Sleep Tracking | 2026 | Calibration |
| 24 | Geenjaar et al., Robust PPG FM | 2026 | Foundation Model |
| 25 | Kjaer et al., BCG-FM | 2026 | Foundation Model |
| 26 | Tackney et al., Missing Accelerometer Framework | 2021 | Imputation |
| 27 | Singh et al., DEM | 2026 | Missing Data |

---

*Report generated for RL Health Interventions research landscape scan, Mission 6 of 10.*
