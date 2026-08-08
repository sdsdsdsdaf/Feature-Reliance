# SAE latent-steering 논문 목록

자동 수집 후 venue/year를 독립 검증한 결과. 원시 90건 -> 중복 제거 88건.
`*` = 검증 단계에서 venue가 정정된 항목.

## architecture (13)

| | Year | Venue | Title | First author | arXiv |
|---|---|---|---|---|---|
|  | 2026 | AAAI 2026 (Student Abstract), Proc. AAAI Conf. on AI Vol. 40 No. 48, pp. 41263-41265 | Steering Sparse Autoencoder Latents to Control Dynamic Head Pruning in Vision Transformers (Student Abstract) | Yousung Lee | 2603.26743 |
|  | 2026 | ICLR 2026 Workshop (Principled Design for Trustworthy AI) | SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks | Vegard Flovik | 2512.15938 |
|  | 2026 | ICML 2026 Workshop (Mechanistic Interpretability), Spotlight | Size Doesn't Matter: Cosine-Scored Sparse Autoencoders | Silen Naihin | 2606.15054 |
|  | 2025 | ICLR 2025 (Oral) | Scaling and evaluating sparse autoencoders | Leo Gao | 2406.04093 |
|  | 2025 | ICML 2025 (PMLR v267) | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | Bart Bussmann | 2503.17547 |
|  | 2025 | Transformer Circuits Thread (non-archival) | Circuits Updates — January 2025: Dictionary Learning Optimization Techniques | Tom Conerly |  |
|  | 2024 | AI Alignment Forum / LessWrong (non-archival blog post, not peer-reviewed) | Addressing Feature Suppression in SAEs | Benjamin Wright |  |
|  | 2024 | BlackboxNLP 2024 (7th BlackboxNLP Workshop, EMNLP) | Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2 | Tom Lieberum | 2408.05147 |
| * | 2024 | NeurIPS 2024 | Improving Dictionary Learning with Gated Sparse Autoencoders | Senthooran Rajamanoharan | 2404.16014 |
| * | 2024 | NeurIPS 2024 Workshop (SciForDL) | BatchTopK Sparse Autoencoders | Bart Bussmann | 2412.06410 |
|  | 2024 | Transformer Circuits Thread (non-archival) | Circuits Updates — February 2024: Update on Dictionary Learning Improvements | Adly Templeton |  |
|  | 2024 | Transformer Circuits Thread (non-archival) | Circuits Updates — April 2024: Update on how we train SAEs / Scaling Laws for Dictionary Learning | Tom Conerly |  |
|  | 2024 | arXiv preprint | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders | Senthooran Rajamanoharan | 2407.14435 |

**Steering Sparse Autoencoder Latents to Control Dynamic Head Pruning in**
- steering: k-SAE trained on the final-layer residual embedding of a ViT; selected sparse latents are amplified under several strategies to alter dynamic attention-head pruning decisions, with per-class steering exposing compact class-specific head subsets that preserve accuracy.
- normalization: Not reported in the abstract/metadata; the 3-page student abstract does not document pre-SAE activation normalization.

**SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Contro**
- steering: Learns an unsupervised sparse feature basis with an l1-regularized autoencoder, validates features with Grad-FAM saliency, then edits the latent vectors and propagates the edit into weight-space modifications; introduces a 'critical suppression threshold' metric for how far a feature must be suppressed before behavior changes. Tested on both CNNs and transformers.
- normalization: Not reported; the method is l1-regularized rather than TopK/JumpReLU, so no unit-norm or scale convention is stated at abstract level.

**Size Doesn't Matter: Cosine-Scored Sparse Autoencoders**
- steering: No steering; changes how z is computed. Encoder score s_i(x) = e^b * ||x_c||^a * cos(x_c, w_i) + b_enc,i with x_c = x - b_dec, unit-normalized encoder rows w_i, and a learned exponent a interpolating between pure cosine (a=0) and inner product (a=1); the fitted a averages ~0.26 and never exceeds 0.5.
- normalization: Directly about activation-scale normalization: because RMSNorm/LayerNorm means each sublayer reads x/||x|| up to a per-coordinate gain, the standard inner-product encoder 'detects a quantity the downstream computation ignores', and residual-stream norms are heavy-tailed (rogue/outlier dimensions), so the resulting magnitude bias is not hypothetical. Encoder rows are unit-normalized and activations are mean-centered by b_dec before scoring; the magnitude dependence is a learned per-feature exponent rather than a fixed preprocessing choice.

**Scaling and evaluating sparse autoencoders**
- steering: No steering. z is hard-sparsified by TopK (keep k largest pre-activations, zero the rest). For evaluation they zero-ablate individual latents and measure downstream logit effects ('ablation sparsity'), and substitute the full reconstruction for loss-recovered.
- normalization: THE most aggressive normalization of this set, and it is PER-TOKEN: 'We subtract the mean over the d_model dimension and normalize to all inputs to unit norm, prior to passing to the autoencoder (or computing reconstruction errors).' Decoder latent directions scaled to unit norm at init AND after every training step (App. C.4: optional for TopK since there is no L1 to game, but 'we find that it still improves MSE, so we still use it'), with Adam gradients projected orthogonal to decoder columns. Loss normalization: main MSE uses ONE normalization constant computed at the start of training, no per-batch normalization; the AuxK MSE is normalized per token 'because the scale of the error changes throughout training'. No latent normalization. They explicitly frame TopK as removing L1 shrinkage.

**Learning Multi-Level Features with Matryoshka Sparse Autoencoders**
- steering: No steering. z is sparsified by BatchTopK; the novelty is that nested prefixes z_{0:m} must each reconstruct x independently. For evaluation they ablate class-specific latents (SCR / Targeted Probe Perturbation) to test disentanglement.
- normalization: NO input-activation normalization and NO decoder unit-norm constraint stated for the LM experiments (they inherit BatchTopK, which also states none). Loss = sum over nested prefixes m in M of ||x - (f(x)_{0:m} W_dec^{0:m} + b_dec)||_2^2 + alpha*L_aux — un-normalized MSE, equally weighted across prefixes, with no 1/|M| and no ||x|| division (Appendix G.1 ablates weighted variants). The only normalization anywhere is in the synthetic toy model: 'l1 sparsity penalty on normalized activations' (method unspecified).

**Circuits Updates — January 2025: Dictionary Learning Optimization Tech**
- steering: No steering; SAE training-recipe note.
- normalization: Most explicit Anthropic statement, verbatim: 'The dataset is scaled by a single constant such that E_{x in X}[||x||_2] = sqrt(n).' Sparsity penalty is tanh-shaped AND decoder-norm-weighted: lambda_S * sum_i tanh(c * |f_i(x)| * ||W_{d,i}||_2) with c = 4 — same gradient as L1 near the activation boundary but zero penalty for strongly-active latents, so it removes the shrinkage incentive. They define a normalized model with W_d' = W_d/||W_d||_2 (unit-norm columns) into which the latent scale is absorbed, i.e. the effective latent is f_i*||W_{d,i}||_2 rather than f_i. Plus a small pre-activation penalty on non-firing latents to reduce dead features.

**Addressing Feature Suppression in SAEs**
- steering: Post-hoc latent rescaling: after SAE training, freeze W_enc/W_dec and fine-tune a learned per-latent scale vector applied to z to undo L1 shrinkage.
- normalization: The canonical shrinkage/'feature suppression' analysis: the L1 penalty plus the unit-norm decoder constraint makes the loss-optimal activation systematically undershoot the true activation (Lasso-style bias), and the effect is worse for high-norm inputs. Their fix is latent-side, not input-side — a per-latent multiplicative scale learned after training. Rajamanoharan et al. note this rescaling alone 'is not necessarily enough' to close the gap. Included because it is the reference every architecture paper (Gated, TopK, JumpReLU, tanh-penalty) cites when justifying its scale handling, despite not being an arXiv paper.

**Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma **
- steering: No steering method proposed; z is JumpReLU-thresholded. The paper lists SAE feature steering vs. steering vectors as open problems for users of the released SAEs.
- normalization: The clearest statement of the convention in the literature: 'During training, activation vectors are normalized by a fixed scalar to have unit mean squared norm' — a single DATASET-level scalar, not per-token — explicitly so that lambda and the bandwidth eps transfer across layers/sites, 'as the raw activation norms can vary over multiple orders of magnitude'. Footnote: 'This is similar in spirit to Conerly et al. (2024), who normalize the dataset to have mean norm of sqrt(d_model).' W_dec columns restricted to unit norm by renormalizing after every update, with gradient components parallel to the columns projected out; W_dec init He-uniform then rescaled to unit norm, W_enc = W_dec^T at init (untied afterwards); theta init to 0.001; b_enc, b_dec init to zero. CRITICAL for downstream users: 'Once training is complete, we rescale the trained SAE parameters so that no input normalization is required for inference' (Appendix A folds both the normalization scalar and the pre-encoder bias into the released weights). For transcoders they 'divide the input and target activations by the root mean square of the input activations'.

**Improving Dictionary Learning with Gated Sparse Autoencoders**
- steering: No steering. z = f(x) is used only for reconstruction; latents are zero-ablated wholesale for the loss-recovered metric, and a diagnostic 'baseline + rescale & shift' learns a per-latent scale r_mag and shift on a frozen SAE to isolate how much of the gap is shrinkage.
- normalization: NO input-activation normalization. Decoder columns constrained to EXACTLY unit norm every step (they note Templeton et al. suggest <=1 but keep =1 'for the sake of simplicity'). Appendix D.1.1: 'In our infrastructure we calculate L2 loss and then divide by n. In the baseline experiments we further divide the reconstruction L2 loss by E||x||_2' — done for Pythia-2.8B/Gemma-7B baselines 'motivated by better hyperparameter transfer' (and NOT for GELU-1L). No latent normalization, but the Gated architecture itself carries a learned per-latent rescaling vector r_mag in R^M.

**BatchTopK Sparse Autoencoders**
- steering: No steering. z is sparsified by taking the B*K largest activations across the WHOLE batch (not per-token); at inference BatchTopK is swapped for a fixed global threshold so behaviour is per-sample and consistent.
- normalization: NO activation normalization, NO decoder unit-norm constraint, and no loss normalization are stated anywhere in the paper — unit-norm decoders appear only when describing prior work's fix for L1 scale-gaming. Loss = ||X - BatchTopK(W_enc X + b_enc)W_dec + b_dec||_2^2 + alpha*L_aux (AuxK from Gao et al.), with no division by ||x||. The only scale machinery is latent-side: the inference threshold theta = E_X[min{z_ij(X) | z_ij(X) > 0}], i.e. the average smallest positive activation over batches.

**Circuits Updates — February 2024: Update on Dictionary Learning Improv**
- steering: No steering; SAE training-recipe note.
- normalization: Normalize activation vectors to have L2 norm equal to sqrt(n_dense), and take the SUM (not the mean) over the dense dimension in the MSE loss — the stated purpose is that hyperparameters then generalize across model sizes. Decoder norm constraint RELAXED from exactly 1 to <= 1, which lets them prune features whose decoder norm is below 0.99 after training.

**Circuits Updates — April 2024: Update on how we train SAEs / Scaling L**
- steering: No steering; SAE training-recipe and scaling-law note.
- normalization: This is the 'Conerly et al. (2024)' recipe everyone cites. Two components: (a) the dataset is scaled by a single constant to mean norm sqrt(d_model), and (b) the L1 term is modified to weight each latent by its decoder column norm, lambda * sum_i f_i(x)*||W_dec,i||_2, which REMOVES the need for a hard unit-norm decoder constraint (Gao et al. call this 'a modified L1 term, as in Conerly et al. [2024]', necessary 'because otherwise the L1 loss can be gamed by making the latents arbitrarily small'). Scaling-law sweeps use MSE + 5*L1. CAVEAT: the transformer-circuits page would not render as text for me; the sqrt(d_model) scaling is confirmed via the Gemma Scope footnote citing it and via the January 2025 update restating it verbatim, not from the April page itself.

**Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse **
- steering: No steering; pure training-method paper. z is thresholded elementwise by JumpReLU (z_i = a_i * 1[a_i > theta_i]) with straight-through estimators on both the L0 penalty and the threshold.
- normalization: YES, dataset-level scalar normalization. Appendix I: 'We normalise LM activations so that they have mean squared L2 norm of one during SAE training. This helps to transfer hyperparameters between different models, sites and layers.' The KDE bandwidth eps=0.001 is explicitly calibrated 'assuming a dataset normalised such that E_x[x^2]=1'. NO decoder unit-norm constraint — they use the reparameterisation-invariant RI-L1 penalty S(f) = sum_i f_i ||d_i||_2 (from Conerly et al. 2024), 'making it unnecessary to impose constraints on ||d_i||_2'. Loss = ||x - x_hat||_2^2 + lambda*||f(x)||_0, with NO division by ||x||.

## vision (14)

| | Year | Venue | Title | First author | arXiv |
|---|---|---|---|---|---|
|  | 2026 | CVPR 2026 | Interpretable and Steerable Concept Bottleneck Sparse Autoencoders | Akshay Kulkarni | 2512.10805 |
|  | 2026 | CVPR 2026 | Language Models Can Explain Visual Features via Steering | Javier Ferrando | 2603.22593 |
| * | 2026 | CVPR 2026 Workshop (Findings of CVPR) | SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models | Quentin Guimard | 2603.19028 |
|  | 2026 | ICML 2026 | Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers (ViSAE) | Tang Li | 2606.06664 |
|  | 2026 | WACV 2026 | SAVE: Sparse Autoencoder-Driven Visual Information Enhancement for Mitigating Object Hallucination | Sangha Park | 2512.07730 |
| * | 2025 | CVPR 2025 Workshop (MIV) | Steering CLIP's Vision Transformer with Sparse Autoencoders | Sonia Joseph | 2504.08729 |
|  | 2025 | Findings of EMNLP 2025 | Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation | Zhenglin Hua | 2505.16146 |
|  | 2025 | ICCV 2025 | SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders | Jiahui Geng (ICCV camera-ready author order; arXiv v1 lists Qing Li first) | 2503.14530 |
|  | 2025 | ICLR 2025 | Sparse autoencoders reveal selective remapping of visual concepts during adaptation (PatchSAE) | Hyesu Lim | 2412.05276 |
|  | 2025 | ICML 2025 | Interpreting CLIP with Hierarchical Sparse Autoencoders (MSAE) | Vladimir Zaigrajew | 2502.20578 |
|  | 2025 | NeurIPS 2025 | Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models | Mateusz Pach | 2504.02821 |
|  | 2025 | arXiv preprint | Beyond Interpretability: When, Why, and How Sparse Autoencoders Enable Label-Free Visual Steering (VS2) | Gerasimos Chatzoudis | 2506.01247 |
|  | 2025 | arXiv preprint | Interpretable and Testable Vision Features via Sparse Autoencoders | Samuel Stevens | 2502.06755 |
|  | 2024 | NeurIPS 2024 | Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE) | Usha Bhalla | 2402.10376 |

**Interpretable and Steerable Concept Bottleneck Sparse Autoencoders**
- steering: Two protocols: 'Unit Vector' — target neuron set to alpha=50, all others to 0; 'White Image' — target neuron set to alpha=50, others set to their white-image values; decode through the SAE and feed into the LVLM. CB-SAE prunes low-utility neurons and adds a concept-bottleneck aligned to a user-defined concept set.
- normalization: Shared bias subtracted inside the SAE: z = sigma_sae(E_sae(v - b)). No explicit input standardization described. Backbones: CLIP ViT-L/14-336 (primary), DINOv2-Large, plus ViT-B/16, ViT-L/14, SigLIP, DFN, PE in sensitivity analysis.

**Language Models Can Explain Visual Features via Steering**
- steering: do(m_sub^l(I_tilde) <- m_sub^l(I_tilde) + alpha * W_dec[i,:]) -- inject a single SAE decoder row across all spatial positions of the vision-encoder residual stream while feeding an EMPTY (white) image, then have the LM verbalize what it sees. alpha tuned on a 500-feature validation set. Also proposes Steering-informed Top-k, a hybrid of causal steering and input-example explanations.
- normalization: TopK activation enforces sparsity; no normalization of the injected decoder vectors and no pre-SAE activation normalization stated.

**SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Lang**
- steering: Multiplicative modulation rather than zeroing: h_debias = h_q ⊙ M + (1-M) ⊙ m_div, with per-neuron coefficients M(j) = S_concept(j)^2 (bias-agnostic) or (1 + S_concept(j) - S_bias(j))^2 (bias-aware), attenuating bias-relevant latents while preserving query-relevant ones. CAVEAT: the Matryoshka SAE is trained on CLIP TEXT embeddings (8.5M CC12M captions), not on vision activations.
- normalization: Only a learned centering parameter b_pre subtracted before encoding and added back after decoding. No additional L2 normalization or mean-centering of embeddings documented.

**Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for In**
- steering: Concept editing via activation patching on per-layer SAE latents (two SAEs per layer: one for CLS, one for image tokens, on CLIP ViT-B/32 residual stream) — set an undesired concept latent to zero (or amplify a desired one) and reconstruct through the decoder; ablating 'grass'/'land' lifts WaterBirds worst-group accuracy by 48.2%.
- normalization: Decoder columns constrained to unit L2 norm. No explicit pre-encoder input standardization described.

**SAVE: Sparse Autoencoder-Driven Visual Information Enhancement for Mit**
- steering: x_steered = x + alpha * W_dec[j,:] where j is a 'visual understanding' latent found via a binary object-presence QA probe. For LLaVA-NeXT and Qwen2-VL a variant operates on the encoder output before TopK. Layer-dependent strength: alpha=3 at early layers (8,12), alpha in {3,5} mid, alpha in {5,10,15} deep.
- normalization: Explicitly checked and ABSENT: no unit-norm decoder vectors, no normalization of hidden states before SAE encoding, no norm-preservation after steering. SAE trained with plain reconstruction loss + sparsity regularization.

**Steering CLIP's Vision Transformer with Sparse Autoencoders**
- steering: 'Select a feature f and replace its feature activation across all patches with a steering strength s' during the forward pass, then decode; for the disentanglement tasks (CelebA, Waterbirds, typographic attacks) they instead zero-ablate the identified spurious latent set per layer. ~10-15% of features are steerable, but SAEs expose thousands more steerable knobs than neurons.
- normalization: Feature vectors normalized to unit length in their steerability metric (Eq. 3); training details list only L2 reconstruction + L1 - no activation normalization at SAE training time.

**Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation**
- steering: SSL adds alpha*d_faithful to visual-token residuals during visual fusion and subtracts alpha*d_hall from generated-token residuals during decoding, with alpha = gamma * ||x_residual|| / (||d_steer|| + eps).
- normalization: Strongest normalization statement in the set: norm-ADAPTIVE steering. The coefficient is explicitly the ratio of residual-stream norm to steering-direction norm, so intervention magnitude scales with local activation norm. No pre-SAE activation normalization or unit-norm decoder stated.

**SAUCE: Selective Concept Unlearning in Vision-Language Models with Spa**
- steering: At inference, scale target-concept latents z_i (i in F_c) by a NEGATIVE factor gamma (gamma = -0.5 in experiments), leaving all other latents untouched, then decode -- selective suppression rather than full ablation.
- normalization: Encoder subtracts a pre-bias (x - b_pre) then ReLU; standard reconstruction loss with no documented normalization layer, no activation standardization, no unit-norm decoder.

**Sparse autoencoders reveal selective remapping of visual concepts duri**
- steering: Binary mask over the top-k SAE latent activation vector before the decoder (keep/kill chosen latents), then the CLIP ViT representation is replaced by the masked SAE reconstruction to measure the causal effect on class logits.
- normalization: Encoder operates on centered activations: SAE(z - b_dec), i.e. a learned pre/decoder bias is subtracted. No input standardization (unit-norm / sqrt(d) rescaling) reported.

**Interpreting CLIP with Hierarchical Sparse Autoencoders (MSAE)**
- steering: Directly set/scale a single concept coefficient in SAE space (e.g. 'germany' from 0.3 -> 20 -> 30; 'bearded'/'glasses'/'blonde' on CelebA) and map back to CLIP space via x_hat = W_dec z + b_pre, then observe shifts in nearest neighbors and classifier probabilities.
- normalization: Most detailed normalization recipe: (1) subtract the per-modality mean estimated on the training set, (2) rescale so E_x[||x||_2] = sqrt(n) with n the embedding dim, (3) unit-norm decoder columns. Training statistics come from the IMAGE modality; at inference on text they swap in the text-specific mean and scaling factor. Operates on post-pooled CLIP ViT-L/14 embeddings (768-d), trained on CC3M.

**Sparse Autoencoders Learn Monosemantic Features in Vision-Language Mod**
- steering: Replace the k-th latent's activation with a constant alpha across all token embeddings (all other latents unchanged) and decode; applied inside CLIP's vision encoder this steers a downstream multimodal LLM (LLaVA) with no change to the language model.
- normalization: No activation normalization at SAE training (plain ReLU + L2 reconstruction + L1). Min-max normalization of activations is used only when computing their monosemanticity (MS) evaluation metric.

**Beyond Interpretability: When, Why, and How Sparse Autoencoders Enable**
- steering: Amplify the input's OWN active top-k latents by gamma>1, form v = Dec(gamma*c) - Dec(c), then x_hat = (x + lambda*v) * ||x||_2 / ||x + lambda*v||_2 on the final-layer CLS token of a frozen CLIP encoder; equivalent to centroid-deviation steering.
- normalization: Learned pre-bias b_pre initialized to the empirical activation mean; encoder acts on (x - b_pre) with TopK. No further input standardization. Note the steering step re-normalizes the steered embedding back to the ORIGINAL L2 norm, which matters for CLIP's cosine-similarity head. Earlier title of this preprint was 'Visual Sparse Steering: Improving Zero-shot Image Classification with Sparsity Guided Steering Vectors'.

**Interpretable and Testable Vision Features via Sparse Autoencoders**
- steering: Encode ViT activations, cache the reconstruction error, suppress or amplify individual latent dimensions in f(x), decode, ADD THE CACHED ERROR BACK, then push the edited activations through frozen task heads (classification/segmentation) and compare outputs. Models: CLIP ViT-B/16 and DINOv2 ViT-B/14.
- normalization: Most explicit of the set: 'we subtract the mean activation vector and normalize activation vectors to unit norm' before SAE training, and decoder columns are re-normalized to unit length after every gradient update. ReLU SAE with L1 penalty, L0 tracked for model selection. NOTE: earlier versions of this arXiv entry were titled 'Sparse Autoencoders for Scientifically Rigorous Interpretation of Vision Models' (the saev package).

**Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)**
- steering: Not an SAE but nonnegative sparse recovery over a text-derived concept dictionary; intervention = zero out the weight on selected concepts in the decomposition (or ablate those probe weights) and re-project back to CLIP space.
- normalization: Explicit and load-bearing: CLIP image embeddings are mean-centered with the image-cone mean estimated on MSCOCO, then RE-NORMALIZED to the unit sphere ('embeddings need to be re-normalized after centering to ensure they lie on the unit-sphere'); the concept dictionary gets the identical centering + normalization.

## LLM (42)

| | Year | Venue | Title | First author | arXiv |
|---|---|---|---|---|---|
| * | 2026 | AAAI 2026 | Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal | Nirmalendu Prakash | 2509.09708 |
|  | 2026 | ACL 2026 (Volume 1: Long Papers) | CRISP: Persistent Concept Unlearning via Sparse Autoencoders | Tomer Ashuach | 2508.13650 |
|  | 2026 | ACL 2026 (Volume 1: Long Papers) | Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models | Ruixuan Deng | 2506.18141 |
|  | 2026 | ACL 2026 (Volume 1: Long Papers) | Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation | Dianyun Wang | 2512.23260 |
|  | 2026 | Findings of EACL 2026 | Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering | Haiyan Zhao | 2505.15038 |
|  | 2026 | ICLR 2026 | ActivationReasoning: Logical Reasoning in Latent Activation Spaces | Lukas Helff | 2510.18184 |
|  | 2026 | ICLR 2026 Workshop (Agentic AI in the Wild: From Hallucinations to Reliable Autonomy) | Steering Large Language Models Toward Clarification through Sparse Autoencoders | Alisa Petrova |  |
|  | 2026 | ICLR 2026 Workshop (Principled Design for Trustworthy AI) | Control Reinforcement Learning: Interpretable Token-Level Steering of LLMs via Sparse Autoencoder Features | Seonglae Cho | 2602.10437 |
|  | 2026 | ICML 2026 | CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features | Seonglae Cho | 2508.12535 |
|  | 2025 | ACL 2025 (Long Papers) | Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms | Mengru Wang | 2505.20322 |
|  | 2025 | ACL 2025 (Volume 1: Long Papers) | Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders | Boyi Deng | 2505.05111 |
|  | 2025 | ACL 2025 (Volume 1: Long Papers) | Sparse Latents Steer Retrieval-Augmented Generation | Chunlei Xin |  |
| * | 2025 | COLM 2025 | Steering Large Language Model Activations in Sparse Spaces | Reza Bayat | 2503.00177 |
| * | 2025 | COLM 2025 | SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs | Aashiq Muhamed | 2504.08192 |
|  | 2025 | EMNLP 2025 (Main Conference) | SAEs Are Good for Steering - If You Select the Right Features | Dana Arad | 2505.20063 |
|  | 2025 | EMNLP 2025 (Main Conference) | SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models | Zirui He | 2505.16188 |
|  | 2025 | EMNLP 2025 (Main Conference) | Toward Efficient Sparse Autoencoder-Guided Steering for Improved In-Context Learning in Large Language Models | Ikhyun Cho |  |
|  | 2025 | EMNLP 2025 (Main Conference) | Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders | Dong Shu | 2505.08080 |
|  | 2025 | EMNLP 2025 (Main Conference) | Model Unlearning via Sparse Autoencoder Subspace Guided Projections | Xu Wang | 2505.24428 |
|  | 2025 | EMNLP 2025 (Main Conference) | LinguaLens: Towards Interpreting Linguistic Mechanisms of Large Language Models via Sparse Auto-Encoder | Yi Jing | 2502.20344 |
|  | 2025 | Findings of EMNLP 2025 | Improving LLM Reasoning through Interpretable Role-Playing Steering | Anyi Wang | 2506.07335 |
|  | 2025 | Findings of EMNLP 2025 | Understanding Refusal in Language Models with Sparse Autoencoders | Wei Jie Yeo | 2505.23556 |
|  | 2025 | Findings of NAACL 2025 | Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting Rare Concepts in Foundation Models | Aashiq Muhamed | 2411.00743 |
|  | 2025 | ICLR 2025 | Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models | Samuel Marks | 2403.19647 |
| * | 2025 | ICLR 2025 | Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control | Aleksandar Makelov | 2405.08366 |
| * | 2025 | ICLR 2025 Workshop (BuildingTrust) | Interpretable Steering of Large Language Models with Feature Guided Activation Additions | Samuel Soo | 2501.09929 |
|  | 2025 | ICML 2025 | SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability | Adam Karvonen | 2503.09532 |
|  | 2025 | ICML 2025 | Analyze Feature Flow to Enhance Interpretation and Steering in Language Models | Daniil Laptev | 2502.03032 |
| * | 2025 | ICML 2025 | Scaling Sparse Feature Circuits For Studying In-Context Learning | Dmitrii Kharlapenko |  |
|  | 2025 | ICML 2025 (spotlight) | AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | Zhengxuan Wu | 2501.17148 |
|  | 2025 | ICML 2025 Workshop (Actionable Interpretability) | Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts | Shruti Joshi | 2502.12179 |
|  | 2025 | ICML 2025 Workshop (Actionable Interpretability) | Resilient Multi-Concept Steering in LLMs via Enhanced Sparse "Conditioned" Autoencoders | Saurish Srivastava |  |
|  | 2025 | ICML 2025 Workshop (GenBio — 2nd Workshop on Generative AI and Biology) | Sparse Autoencoders in Protein Engineering Campaigns: Steering and Model Diffing | Gerard Corominas |  |
|  | 2025 | NAACL 2025 (Long Papers; oral) | Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering | Yu Zhao | 2410.15999 |
|  | 2024 | Anthropic research post (non-peer-reviewed) | Evaluating Feature Steering: A Case Study in Mitigating Social Biases | Esin Durmus |  |
|  | 2024 | ICLR 2024 (poster) | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Hoagy Cunningham (ICLR proceedings list Robert Huben first) | 2309.08600 |
|  | 2024 | ICML 2024 Workshop (MI), Spotlight | Sparse Autoencoders Match Supervised Features for Model Steering on the IOI Task | Aleksandar Makelov |  |
|  | 2024 | NeurIPS 2024 Workshop (Safe Generative AI) | Applying Sparse Autoencoders to Unlearn Knowledge in Language Models | Eoin Farrell | 2410.19278 |
|  | 2024 | Transformer Circuits Thread (Anthropic, non-peer-reviewed) | Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet | Adly Templeton | 2605.29358 |
|  | 2024 | arXiv preprint | Improving Steering Vectors by Targeting Sparse Autoencoder Features | Sviatoslav Chalnev | 2411.02193 |
|  | 2024 | arXiv preprint | Steering Language Model Refusal with Sparse Autoencoders | Kyle O'Brien | 2411.11296 |
|  | 2023 | Transformer Circuits Thread (Anthropic, non-peer-reviewed) | Towards Monosemanticity: Decomposing Language Models With Dictionary Learning | Trenton Bricken |  |

**Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal**
- steering: Search the SAE latent space for a minimal set of refusal-mediating latents and ablate them (remove their contribution) to flip Gemma-2-2B-IT / Llama-3.1-8B-IT from refusal to compliance; candidates are seeded from a refusal direction, pruned greedily, then re-ranked with a factorization machine that models non-linear latent interactions.
- normalization: No activation-normalization statement.

**CRISP: Persistent Concept Unlearning via Sparse Autoencoders**
- steering: Not inference-time steering: LoRA fine-tuning with L_unlearn = E_t E_{f_i in F_salient}[a_i^(t) + lambda*c_t], driving salient SAE feature activations toward zero on target data, combined with retain and coherence losses so suppression is persistent in the weights.
- normalization: Operates on RAW SAE feature activations with no preprocessing normalization. The only normalization is of downstream evaluation metrics (fluency/concept scores rescaled to 0-100).

**Sparse Feature Coactivation Reveals Causal Semantic Modules in Large L**
- steering: Group coactivating SAE features into 'components', then ablate (set activations to zero) or amplify by raising each feature by a proportion alpha of its maximum observed activation, replacing layer activations with the SAE decoder output before continuing the forward pass.
- normalization: No explicit activation/latent normalization. Notably, amplification is scaled RELATIVE TO EACH FEATURE'S OWN MAX OBSERVED ACTIVATION rather than a global norm -- a per-feature scale calibration in place of normalization.

**Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace A**
- steering: SAILS builds an interpretable safety subspace from SAE decoder directions and uses it to initialize LoRA adapters -- weight-space steering via SAE-derived directions rather than inference-time latent editing (0.19% of parameters updated).
- normalization: Not stated in the abstract; unverified (only the abstract and accepted-papers listing were checked).

**Denoising Concept Vectors with Sparse Autoencoders for Improved Langua**
- steering: SDCV reconstructs hidden representations from only the top-k most discriminative SAE latents (scaling those activations up) to denoise linear concept vectors, then applies the denoised vectors with standard linear-probe and difference-in-means steering.
- normalization: None stated on the ACL Anthology page. Note: EACL, adjacent to the requested venue list.

**ActivationReasoning: Logical Reasoning in Latent Activation Spaces**
- steering: Builds a dictionary of SAE latent concept representations, maps detected active latents to logical propositions at inference, applies logical rules to compose higher-order/new concepts, and then activates those composed concept latents to steer behavior (evaluated on PrOntoQA, Rail2Country, ProverQA, BeaverTails).
- normalization: Not stated in the abstract/summary retrieved; concept dictionary construction is SAE-based but no normalization protocol reported at this level of detail.

**Steering Large Language Models Toward Clarification through Sparse Aut**
- steering: ClarifySAE scores and filters SAE features by association with clarification-seeking contexts (ClarifyScore), then applies an additive bias to those features during decoding — inference-time only, no weight updates. Tested on ambiguous-instruction datasets with Gemma models.
- normalization: Not reported on the venue page.

**Control Reinforcement Learning: Interpretable Token-Level Steering of **
- steering: Per the accepted-papers listing, learns a token-level control policy over SAE features via reinforcement learning, so the steering intervention (which feature, what magnitude) is chosen adaptively at each decoding step rather than fixed globally. Same group as CorrSteer.
- normalization: Not available — only the workshop accepted-papers listing was retrievable; no PDF-level detail confirmed.

**CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoenco**
- steering: Selects latents by correlating sample correctness with SAE activations computed on generated (not prompt) tokens, validates by intervention, then adds v_steer = c_i * W_dec[:,i] to the residual stream: x' = x + v_steer. No clamping.
- normalization: No explicit training-time normalization (uses released Gemma Scope JumpReLU SAEs). The steering coefficient is implicitly scale-anchored: c_i = mean of z_{i,j} over samples with positive outcome, i.e. the feature's own natural activation magnitude.

**Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steerin**
- steering: Steering Target Atoms (STA): select latents whose activation-amplitude difference (delta a) and activation-frequency difference (delta f) between positive/negative sets exceed thresholds, map the selected sparse vector back through the decoder v_STA = a_target W_dec + b_dec, then add h_hat = lambda*v_STA + h. Additive, not clamping.
- normalization: Steering vectors from competing methods are rescaled to equal magnitude for fair comparison; no SAE-training normalization statement.

**Unveiling Language-Specific Features in Large Language Models via Spar**
- steering: Directional ablation of language-specific SAE features (x <- x - d_hat d_hat^T x), plus a gated steering vector added to the residual stream only when the top-2 target-language SAE features are non-zero.
- normalization: No activation normalization scheme stated; decoder columns not explicitly declared unit-norm; steering vectors are not normalized. Confirmed absent in arXiv HTML v2.

**Sparse Latents Steer Retrieval-Augmented Generation**
- steering: Identify SAE latents that govern context-vs-memory prioritization and answer-vs-reject, then manipulate those latents at inference; the causal path is a reconfiguration of retrieval-head attention patterns.
- normalization: None stated on the ACL Anthology page. No arXiv preprint found.

**Steering Large Language Model Activations in Sparse Spaces**
- steering: Sparse Activation Steering (SAS): build v = mean(positive sparse codes) - mean(negative sparse codes) after frequency-filtering and removing shared latents, then at inference a_tilde = decode(sigma(f(a) + lambda*v)) + Delta - i.e. add lambda*v inside the latent code, re-apply the nonneg activation, decode, and add a reconstruction-error correction term.
- normalization: No normalization of activations/latents discussed; a correction term Delta compensates SAE reconstruction error (not a normalization).

**SAEs Can Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for**
- steering: DSG: conditional clamping - a sequence-level classifier fires when rho(x) (fraction of tokens activating any selected forget-latent) exceeds a percentile threshold tau; if so, set f_j'(h_t) = -c for the selected latents (clamp strength c swept 10-500), else pass activations through unchanged.
- normalization: No explicit activation-normalization statement (loss described only as reconstruction + sparsity).

**SAEs Are Good for Steering - If You Select the Right Features**
- steering: For the target latent i: a_tilde_i = a_i + s * a_max (all other latents untouched), then decode through the SAE; the contribution is a selection criterion ('output score') that filters input-pattern latents in favor of output-effect latents, giving 2-3x better steering and making SAEs competitive with supervised steering.
- normalization: Steering step is normalized by the latent's recorded max activation a_max; no SAE-training normalization described.

**SAE-SSV: Supervised Steering in Sparse Representation Spaces for Relia**
- steering: Initialize v_init = mu+ - mu- in SAE latent space, zero all coordinates outside a linear-probe-selected top-d_steer subspace, then optimize L_steer = ||z'-mu+||^2 - ||z'-mu-||^2 + L_LM + beta||v_I||_1.
- normalization: The steering vector IS normalized after zeroing/truncation to the selected subspace (norm type unspecified). No unit-norm decoder constraint and no pre-SAE activation normalization reported; sparsity is enforced by L1 rather than a norm constraint.

**Toward Efficient Sparse Autoencoder-Guided Steering for Improved In-Co**
- steering: FDPV selects SAE features from activation differences across prompt variations; SISTER then applies the steering signal selectively at label-word anchor positions rather than at every token.
- normalization: Not stated on the ACL Anthology landing page; the PDF could not be parsed to confirm. No arXiv preprint located. Treat normalization as unverified.

**Beyond Input Activations: Identifying Influential Latents by Gradient **
- steering: GradSAE ranks latents by output-side gradient information (not just input-side activation) and steers/ablates only the high-influence latents, on the premise that only influential latents are effective for steering.
- normalization: None stated. Paper is about latent selection rather than the intervention operator itself.

**Model Unlearning via Sparse Autoencoder Subspace Guided Projections**
- steering: SSPU is weight-space rather than activation-space: select layer and SAE features, build an 'irrelevant' subspace via QR decomposition of those decoder directions, then constrained optimization pushes activations into that subspace while preserving retained knowledge.
- normalization: QR decomposition yields an orthonormal basis for the projection subspace; no activation normalization or unit-norm decoder reported.

**LinguaLens: Towards Interpreting Linguistic Mechanisms of Large Langua**
- steering: Extract SAE latents for linguistic features across six levels (phonetics, phonology, morphology, syntax, semantics, pragmatics), then intervene on them using minimal-contrast and counterfactual sentence datasets to establish causal control over outputs.
- normalization: None stated on the ACL Anthology page.

**Improving LLM Reasoning through Interpretable Role-Playing Steering**
- steering: SRPS extracts latents from role-play prompts, selects the most relevant SAE features by activation pattern, builds a steering vector from them, and injects it into the residual stream with controllable intensity.
- normalization: Not stated in the abstract/metadata; unverified.

**Understanding Refusal in Language Models with Sparse Autoencoders**
- steering: Identify SAE latents that causally mediate refusal, then intervene (amplify/ablate) on those refusal features to change generation, validated across multiple harmful datasets and adversarial jailbreaks.
- normalization: None stated.

**Decoding Dark Matter: Specialized Sparse Autoencoders for Interpreting**
- steering: Train subdomain-specialized SAEs (SSAEs), then ablate the identified spurious-gender latents on Bias in Bios; ablation raises worst-group accuracy by 12.5% over a general-purpose SAE.
- normalization: No activation normalization mentioned; training discussion covers dense-retrieval data selection and Tilted ERM, plus perplexity and L0 metrics only.

**Sparse Feature Circuits: Discovering and Editing Interpretable Causal **
- steering: SHIFT edits behavior by zero-ablating human-judged-irrelevant SAE latents, IE(m;a;x)=m(x|do(a=0))-m(x); circuit faithfulness instead uses mean-ablation (latents set to their position-specific dataset mean).
- normalization: No statement about normalizing activations before SAE training found in the paper.

**Towards Principled Evaluations of Sparse Autoencoders for Interpretabi**
- steering: Sparse controllability test on IOI: greedily solve min over subsets R (remove) and A (add), |R union A| <= k, of ||a_s - sum_{i in R} alpha_i u_i + sum_{i in A} beta_i u_i - a_t||_2 - i.e. subtract the source-attribute latents' decoder contributions and add the counterfactual-target latents', then check whether the model's output flips; compared against supervised feature dictionaries.
- normalization: Only the decoder unit-norm constraint ||(W_dec)_{:,i}||_2 = 1; no input-activation normalization stated. Edit magnitudes are reported via a normalized per-latent 'weight' share of the reconstruction.

**Interpretable Steering of Large Language Models with Feature Guided Ac**
- steering: FGAA: operate in SAE latent space - select desired latents by contrastive/density-based optimization over positive vs negative prompt sets, then decode the selected latents into a single steering vector added to the residual stream; beats CAA, raw SAE-decoder steering and SAE-TS on Gemma-2-2B/9B.
- normalization: None reported; uses off-the-shelf Gemma Scope SAEs.

**SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Languag**
- steering: Two intervention-based metrics: unlearning 'clamps these latents to negative values whenever they activate' (Farrell-style), and spurious-correlation removal (SHIFT/SCR) measures how well zero-ablating a few latents removes an unwanted correlation.
- normalization: Most explicit of the set: 'we first estimate a scalar constant to normalize the activations to have a unit mean squared norm during training, increasing hyperparameter transfer between layers and models. We fold this constant into the weights after training so our SAEs don't require normalized activations.'

**Analyze Feature Flow to Enhance Interpretation and Steering in Languag**
- steering: Purely additive decoder steering, no clamping: activation h_t <- h_t + s . V^T (amplify) and h_t <- h_t + (r-1)(a_t . V^T) (suppress, r=0 removes the feature). Cross-layer flow graphs let them steer the same feature at multiple layers simultaneously, distributing coefficients by exponential decay s' = s * e^(alpha*l) or linear interpolation.
- normalization: For the cross-layer cosine-matching procedure they state: 'We assume that both f and the columns of W_dec^(B) have unit norm.' No statement about input-activation normalization at SAE training time (they use pretrained SAEs); steering magnitude is not normalized by activation scale.

**Scaling Sparse Feature Circuits For Studying In-Context Learning**
- steering: Identifies abstract SAE latents encoding 'which task to execute' and adds their decoder latent vectors into the residual stream to causally induce the task zero-shot; shows ICL task vectors can be approximated as sparse combinations of SAE latents. Scales sparse feature circuits to Gemma-1 2B.
- normalization: Not reported.

**AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoen**
- steering: Single-latent SAE steering: add the latent's decoder atom to the residual stream, x_steer = x + (alpha * m_f) * v_f, where m_f is that latent's max activation and alpha is a swept steering factor; a clamping variant instead pins the latent to alpha*m_f (and adds back the unclamped reconstruction error). Conclusion: SAE steering loses to prompting and to rank-1 ReFT-r1.
- normalization: Steering magnitude is explicitly scaled by the latent's maximum activation m_f (max acts sourced from Neuronpedia / activation distributions), which is the paper's de facto latent-scale normalization; no SAE-training normalization of their own (they use released SAEs).

**Identifiable Steering via Sparse Autoencoding of Multi-Concept Shifts**
- steering: Sparse shift autoencoders (SSAEs) encode the *difference* between paired embeddings into a sparse code, so the learned latents are concept shifts; steering is applied by decoding a chosen sparse shift and adding it to the embedding. Requires only paired observations varying in multiple unknown concepts, no concept labels. Validated on Llama-3.1 embeddings.
- normalization: Not reported on the venue page; the identifiability argument is about the shift representation rather than activation scale.

**Resilient Multi-Concept Steering in LLMs via Enhanced Sparse "Conditio**
- steering: M-SCAR extends the Sparse Conditioned Autoencoder (SCAR) so that designated latents are supervised/conditioned on multiple labeled attributes (e.g. toxicity, style) during SAE training; steering is then done by setting or scaling those conditioned latents, allowing simultaneous multi-attribute control without touching base-model weights.
- normalization: Not reported on the venue page.

**Sparse Autoencoders in Protein Engineering Campaigns: Steering and Mod**
- steering: Trains SAEs on protein language model activations, selects enzyme-relevant candidate features by causal intervention, then 'steer[s] the model by clamping or ablating key SAE features' during sequence generation, which increases predicted enzyme activity. Also does checkpoint model-diffing across RL rounds.
- normalization: Not reported on the venue page.

**Steering Knowledge Selection Behaviours in LLMs via SAE-Based Represen**
- steering: SpARE: pick context-vs-memory functional latents by mutual information, compute per-latent removal z_i^- = min(z_i, z_i^C) and addition z_i^+ = max(z_i^M - z_i, 0), then edit the hidden state h' = h + alpha(-g(z^-) + g(z^+)) with g the SAE decoder; they deliberately do NOT set h' = g(z') to avoid reconstruction loss.
- normalization: None stated; constraints are designed only to keep edited latents non-negative (ReLU-compatible).

**Evaluating Feature Steering: A Case Study in Mitigating Social Biases**
- steering: Add a constant in the direction of the chosen feature to the residual stream of Claude 3 Sonnet, with a 'steering factor' swept over -20..+20; only the +/-5 'sweet spot' steers behavior (e.g. a 'neutrality' feature reduces 9 social-bias dimensions) without wrecking capabilities.
- normalization: Post defers SAE/steering details to Scaling Monosemanticity; no independent normalization statement.

**Sparse Autoencoders Find Highly Interpretable Features in Language Mod**
- steering: Interchange interventions on latent coefficients: x_i' = x_bar_i + sum_{j in F} (c_ij - c_bar_ij) f_j, i.e. patch selected dictionary-feature coefficients from a counterfactual run into the clean run, and greedily ablate features one at a time until performance drops (used to find a minimal causal feature set for IOI).
- normalization: Dictionary matrix M is normalized row-wise (unit-norm feature vectors) so the L1 term cannot be reduced by inflating feature-vector magnitudes. No input-activation rescaling reported.

**Sparse Autoencoders Match Supervised Features for Model Steering on th**
- steering: Edits SAE feature activations to counterfactual values on the IOI task (feature-level activation replacement / sparse edit), and shows the resulting steering matches what supervised ground-truth feature dictionaries achieve. Workshop precursor to the ICLR 2025 'Towards Principled Evaluations' paper.
- normalization: Not reported at workshop-paper level; see the ICLR 2025 version for unit-norm decoder + b_dec-centering details.

**Applying Sparse Autoencoders to Unlearn Knowledge in Language Models**
- steering: Clamp latent to a fixed negative value whenever it fires (leave at 0 otherwise): 'we set the feature activation equal to a fixed negative value if the feature activates'. Single-feature case study effective at about -10 to -20; jointly intervening on ~10-20 latents works best (Gemma-2-2b-it, layer-3 16k SAE). Zero ablation is ineffective; negative scaling is required, and fixed-negative clamping beats negative multiplicative scaling in side effects.
- normalization: No statement about activation normalization.

**Scaling Monosemanticity: Extracting Interpretable Features from Claude**
- steering: Clamp a chosen latent's activation to a fixed value during the forward pass, typically a multiple of its max observed activation (the Golden Gate Bridge feature clamped to ~10x max produces 'Golden Gate Claude'); negative clamps suppress.
- normalization: Explicit: 'As a preprocessing step we apply a scalar normalization to the model activations so their average squared L2 norm is the residual stream dimension, D.' Decoder column norms are folded into the L1 penalty so the unit-normalized decoder columns are the 'feature directions'.

**Improving Steering Vectors by Targeting Sparse Autoencoder Features**
- steering: SAE-TS: fit a linear 'effect approximator' from a residual-stream steering vector to the induced change in every SAE latent, then solve for the vector that maximally raises one target latent with minimal off-target latent change, and add that vector at generation time (SAEs used as the measurement instrument, not the knob).
- normalization: No explicit statement about activation/latent normalization at training time (they use pretrained Gemma Scope SAEs).

**Steering Language Model Refusal with Sparse Autoencoders**
- steering: Clamp the refusal-mediating latent in the sparse code to a constant and decode (higher amplifies, lower dampens); grid search on Phi-3 Mini picked clamp values 10 (best safety/utility balance) and 12 (max refusal, more over-refusal) for feature 22373. Steering improves jailbreak robustness but degrades benchmarks even on safe inputs; conditional (classifier-gated) steering explored in the appendix.
- normalization: No activation-normalization discussion.

**Towards Monosemanticity: Decomposing Language Models With Dictionary L**
- steering: 'Pinned feature sampling': pin/clamp one latent's activation to a fixed high value throughout sampling so generation shifts into that mode (e.g. base64 feature -> base64 text, Arabic-script feature -> Arabic text); also zero-ablation of features to measure causal effect.
- normalization: Confirmed indirectly (source page too large to fetch in full): decoder columns are renormalized to unit norm after every training step - later papers cite this practice as 'following Bricken et al. (2023)', and note it is needed so the L1 penalty cannot be gamed by shrinking latents. I could NOT verify any dataset-level activation rescaling in this write-up; the scalar 'average squared L2 norm = D' rescaling is stated only in the 2024 Scaling Monosemanticity paper.

## diffusion (19)

| | Year | Venue | Title | First author | arXiv |
|---|---|---|---|---|---|
| * | 2026 | AAAI 2026 | TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion Transformers in Image Generation | Victor Shea-Jay Huang | 2503.07050 |
|  | 2026 | ECCV 2026 | Look But Don't Touch with Sparse Autoencoders for Unlearning in Diffusion Models | Enrico Cassano | 2606.31699 |
|  | 2026 | ICML 2026 | SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders | Enrico Cassano | 2509.21379 |
|  | 2026 | ICML 2026 | RAIGen: Rare Attribute Identification in Text-to-Image Generative Models | Silpa Vadakkeeveetil Sreelatha | 2602.06806 |
|  | 2026 | arXiv preprint | CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models | Zhenghao He | 2601.15441 |
|  | 2026 | arXiv preprint | Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models | Calvin Yeung | 2605.27813 |
|  | 2026 | arXiv preprint | Robust and Generalizable Safety Steering for Text-to-Image Diffusion Transformers | Zihao Xue | 2605.30049 |
|  | 2026 | arXiv preprint | Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning | Hyeonjin Kim | 2605.12122 |
|  | 2025 | CVPR 2025 | Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability | Yingdong Shi | 2503.20483 |
|  | 2025 | CVPR 2025 Workshop (Mechanistic Interpretability for Vision) | Interpreting Large Text-to-Image Diffusion Models with Dictionary Learning | Stepan Shabalin | 2505.24360 |
| * | 2025 | ICCV 2025 Workshop (Findings of ICCV) | Probing the Representational Power of Sparse Autoencoders in Vision Models | Matthew Lyle Olson | 2508.11277 |
|  | 2025 | ICML 2025 | SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders | Bartosz Cywinski | 2501.18052 |
|  | 2025 | NeurIPS 2025 | Emergence and Evolution of Interpretable Concepts in Diffusion Models | Berk Tinaz | 2504.15473 |
|  | 2025 | NeurIPS 2025 (poster) | One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models | Viacheslav Surkov | 2410.22366 |
| * | 2025 | OpenReview (ICLR 2026 submission, withdrawn; unpublished) | Steering Diffusion Transformers with Sparse Autoencoders | anonymous (double-blind OpenReview submission; authors not disclosed) |  |
|  | 2025 | arXiv preprint | Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations | Dahye Kim | 2501.19066 |
|  | 2025 | arXiv preprint | SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder | Ronen Kamenetsky | 2510.05081 |
|  | 2025 | arXiv preprint | Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Text-to-Image Diffusion Models | Zhihua Tian | 2503.09446 |
|  | 2025 | arXiv preprint | Model-Agnostic Gender Bias Control for Text-to-Image Generation via Sparse Autoencoder | Chao Wu | 2507.20973 |

**TIDE: Temporal-Aware Sparse Autoencoders for Interpretable Diffusion T**
- steering: A TopK k-SAE per DiT layer of PixArt-alpha (28 layers; penultimate block 27 used for editing); control by erasing/inverting top-K latents, scaling selected activation indices for continuous feature transitions, or wholesale replacement of latent features for global changes.
- normalization: k-SAE with TopK (~5% sparsity, 16d latents = 73,728). Encoder initialized as the decoder transpose with periodic dead-latent revival. Notably they add classifier-free-guidance renormalization at intervention time "to ensure the stability of the norm" — i.e. an explicit post-intervention norm correction rather than an input normalization at training.

**Look But Don't Touch with Sparse Autoencoders for Unlearning in Diffus**
- steering: NEGATIVE RESULT on latent steering: direct manipulation of SAE latents induces out-of-distribution activations and severe visual artifacts, so they instead use SAE activations purely as semantic *detectors* to localize the target object and replace those patch embeddings with object-free ones.
- normalization: Highly relevant: the whole method is motivated by activation-statistics preservation — their embedding-swap "preserves the diffusion model's activation statistics", whereas latent edits push activations OOD. Explicit conclusion that "monosemantic or sparse features are not inherently suitable as control knobs."

**SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse**
- steering: Supervised SAE training with systematic concept labeling enforces one-to-one concept-neuron mappings (feature centralization, no feature splitting), so unlearning reduces to suppressing a single interpretable latent — cutting hyperparameter search by 96.7% and scaling to sequential removal of nine objects.
- normalization: Not stated in the abstract; full-text normalization details not verified.

**RAIGen: Rare Attribute Identification in Text-to-Image Generative Mode**
- steering: Matryoshka Sparse Autoencoders plus a minority metric (activation frequency x semantic distinctiveness) identify neurons encoding underrepresented attributes; those latents are then amplified during generation to surface rare attributes.
- normalization: Not stated in the abstract; not verified. (Matryoshka SAEs use nested dictionary subsets, which changes the effective latent scale per nesting level.)

**CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models**
- steering: Supervised SAE on frozen U-Net activations aligns individual sparse dimensions with named concepts; CASL-Steer shifts activations along the learned concept axis — the authors frame this as a causal probe rather than an editing method.
- normalization: Not stated in the abstract; not verified.

**Residualized Temporal Sparse Autoencoders for Interpreting Diffusion M**
- steering: Fits linear predictors between neighboring denoising timesteps and trains the SAE on the residual (non-linearly-predictable) part of the activation trajectory; residualized decoder directions are mapped back into activation space and injected for qualitative steering on Stable Diffusion 1.5.
- normalization: Not stated in the abstract. The residualization itself is a form of trajectory-level preprocessing (subtracting linearly predictable dynamics) rather than a magnitude normalization.

**Robust and Generalizable Safety Steering for Text-to-Image Diffusion T**
- steering: SAEs built over functionally distinct DiT intervention positions (robustness-aware pretraining routing picks stable sites); inference-time Blend and Repel operations push unsafe activations toward transferred safety manifolds or away from harmful sparse directions, on FLUX.1 Dev and SD 3.5 Large.
- normalization: Directly relevant: they freeze the SAE encoder as a reusable sparse safety dictionary and adapt ONLY the decoder to the target-domain activation manifold, explicitly separating transferable safety features from domain-specific activation geometry (i.e. treating activation scale/geometry mismatch as the transfer problem).

**Disentangled Sparse Representations for Concept-Separated Diffusion Un**
- steering: SAEParate organizes latents into concept-specific clusters via a concept-aware contrastive objective, then suppresses the target cluster's latent features at inference (no weight updates), reducing interference with non-target concepts.
- normalization: Encoder augmented with a GeLU-based nonlinear transformation for expressivity; no activation-normalization statement found in the abstract.

**Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretabil**
- steering: DiffLens trains a k-SAE on diffusion hidden activations (s = TopK(W_enc(h - b_pre))), scores bias features by gradient attribution, selects the top-tau, and intervenes per feature by either scaling s_i <- beta*s_i or shifting s_i <- s_i + beta before decoding.
- normalization: The k-SAE subtracts a learned pre-bias b_pre before encoding and adds it back on decode (input centering). No explicit activation standardization, no unit-norm decoder constraint stated.

**Interpreting Large Text-to-Image Diffusion Models with Dictionary Lear**
- steering: TopK SAEs (and ITDA) trained on residual-stream activations of Flux 1 (Schnell) double- and single-blocks; generation is steered by activation addition of an SAE decoder direction into the residual stream, applied only over a sub-range of denoising steps to avoid artifacts.
- normalization: UNVERIFIED. I could not extract the methods section from the PDF (9.5 MB, hit fetch limits). A secondary summary (alphaXiv) states activations were PCA-whitened/standardized and the SAE trained in whitened space with reconstructions mapped back — treat this as unconfirmed until the PDF is read directly.

**Probing the Representational Power of Sparse Autoencoders in Vision Mo**
- steering: SAE trained on token representations from the penultimate layer of Stable Diffusion's text encoder; steering amplifies the text-encoder output along a target SAE feature direction with a scaling coefficient, combined with classifier-free guidance so generation moves toward that semantic attribute.
- normalization: Not confirmed from the abstract page; full-text methods not verified.

**SAeUron: Interpretable Concept Unlearning in Diffusion Models with Spa**
- steering: Conditional negative scaling of concept latents: f_i(x) <- gamma_c * mu(i,t,D_c) * f_i(x) for i in the selected concept set when f_i(x) exceeds its dataset-average activation, else unchanged; gamma_c < 0 (about -1 for styles, -1 to -30 for objects), scaled by the latent's mean activation on concept samples.
- normalization: The negative multiplier is explicitly normalized by the latent's average activation on concept data - i.e. steering strength is expressed in units of that latent's own activation scale.

**Emergence and Evolution of Interpretable Concepts in Diffusion Models**
- steering: SAEs on a text-to-image diffusion model's intermediate activations; targeted interventions on SAE concept latents at chosen denoising stages — early-stage intervention controls scene composition, middle-stage controls style, late-stage only affects texture.
- normalization: Not stated in the abstract; not verified from the full text.

**One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Mo**
- steering: SAEs trained on the *update* (output) of four cross-attention transformer blocks in SDXL Turbo's U-Net; generation is steered by turning individual SAE features on/off (adding/subtracting the decoder direction into the block update, optionally spatially masked), evaluated on their RIEBench representation-based image-editing benchmark.
- normalization: Codebase (surkovv/sdxl-unbox) is built on openai/sparse_autoencoder: latents = relu(topk(W_enc(x - pre_bias) + b_lat)). Input activations are NOT normalized/standardized — only a learnable pre-encoder bias is subtracted. Decoder columns ARE unit-normalized at init (`unit_norm_decoder_`) and kept unit-norm during training via a gradient-projection adjustment.

**Steering Diffusion Transformers with Sparse Autoencoders**
- steering: Multi-layer steering: the same SAE feature is injected at several DiT layers simultaneously to raise the causal effect of the intervention while suppressing artifacts, with a similarity-based criterion on the residual stream deciding which layers to steer at (Flux / SD3-class DiTs).
- normalization: Could not verify — OpenReview blocked automated fetching (bot challenge). Include only after manual confirmation of authors and venue.

**Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Contr**
- steering: Train a k-SAE on the diffusion model's text-encoder features, then at test time x_steered = x + W_dec(lambda * ENC(x_C)), where x_C is the concept's embedding and lambda (signed) controls strength - adds/subtracts the decoded sparse concept direction to the prompt embedding, no retraining or LoRA.
- normalization: Decoder weights W_dec constrained to unit norm after each update; notably, at inference they run the encoder WITHOUT the TopK activation so latent magnitudes are not truncated (training-time TopK only).

**SAEdit: Token-level control for continuous image editing via Sparse Au**
- steering: BatchTopK SAE trained on frozen T5-XXL output text embeddings; per-token editing via e'_token = S_dec(S_enc(e_tgt) + omega * d_edit) — encode to sparse space, add a scaled attribute direction, decode back — giving continuous, disentangled attribute strength control for the downstream DiT.
- normalization: BatchTopK with auxiliary dead-latent loss (Matryoshka SAE variant in appendix). Edit-direction extraction normalizes the activation-ratio vector as R_norm = R / max(R) before thresholding. No statement found about normalizing the T5 embeddings themselves at SAE training time.

**Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Te**
- steering: SAE doubles as a zero-shot classifier detecting whether the prompt contains a target concept, then permanently deactivates the specific SAE features associated with that concept (celebrity identities, artistic styles, explicit content) without retraining.
- normalization: Not stated in the abstract; not verified.

**Model-Agnostic Gender Bias Control for Text-to-Image Generation via Sp**
- steering: k-SAE pretrained on a gender-bias dataset; a per-profession biased direction is constructed from the sparse latents and suppressed at inference to rebalance generations across SD 1.4/1.5/2.1 and SDXL.
- normalization: Not stated in the abstract; not verified.
