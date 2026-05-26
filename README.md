# ELRIC: A Multi-Dimensional Entropy Estimator for Optimal Information Preservation in Image Compression
# Abstract
In recent years, learned image compression (LIC) methods have substantially outperformed classical
codecs across a wide range of bitrates, yet they remain architecturally single-task: compress a full-
resolution image and reconstruct it. Real deployment scenarios demand far more as content delivery
networks simultaneously require a compressed full-quality bitstream, a lightweight thumbnail, a fast
progressive preview, and a super-resolved output for high-density displays. Addressing all four tasks
with separate models incurs prohibitive redundant computation. In this paper, we propose ELRIC
(Efficient Lowered Resolution Image Compression), a unified single-model architecture that delivers
compression, learned spatial downscaling, super-resolution and progressive preview synthesis from
a single forward pass. A Dual-Path Encoder (DPE) branches one shared stem into a compression
latent and a downscaling feature map at no additional encoding cost. A novel Slice Entropy Model
(SEM) combining Hierarchical Dictionary Attention (HDA), Cross-Scale Feature Fusion (CSFF),
Multi-Scale Channel Attention (MSCA), and Gated Window Context Fusion (GWCF), provides
rich per-slice entropy estimates across 퐾 = 8 coding steps. A curriculum rate-distortion loss with
explicit BPP floor and ceiling barriers coordinates all four tasks throughout training. Extensive
experiments demonstrate that ELRIC simultaneously achieves state-of-the-art compression quality
(BD-Rate of −20.87%, −20.14%, and −24.53% on Kodak, CLIC Pro Valid and Tecnick respectively,
against VTM-22.0), faithful ×4 spatial downscaling (PSNR푑푠 up to 39.48 dB on DIV2K), competitive
super-resolution (PSNR푠푟 up to 40.13 dB on BSD100), and improved progressive preview quality
relative to truncated classical streams. For the sake of research reproducibility and enhancement, our
implementation is freely available at: https://github.com/voxtranslate/ELRIC.
