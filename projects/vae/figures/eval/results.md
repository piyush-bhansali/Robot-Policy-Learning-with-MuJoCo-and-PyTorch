| Checkpoint | β | Total loss | Reconstruction loss | Pixel MSE | KL loss |
|---|---|---|---|---|---|
| checkpoints/vae/vae_beta0_s0.pt | 0 | 108.2981 | 108.2981 | 0.003917 | 24.9466 |
| checkpoints/vae/vae_beta0.5_s0.pt | 0.5 | 25.3919 | 16.3113 | 0.000590 | 18.1611 |
| checkpoints/vae/vae_beta1_s0.pt | 1 | 33.8686 | 19.6296 | 0.000710 | 14.2390 |
| checkpoints/vae/vae_beta4_s0.pt | 4 | 108.3176 | 108.3174 | 0.003918 | 0.0000 |

Split: test, 500 images. Seed: 0.

Metrics use sampled latents; reconstruction figures use latent means.

Reconstruction loss sums squared RGB pixel errors per image, then averages over images. Pixel MSE divides that loss by 27,648. KL is summed over latent dimensions, then averaged over images.

Total loss includes each model's β, so compare reconstruction and KL separately across β cases.
