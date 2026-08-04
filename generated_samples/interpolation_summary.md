# FashionGen Studio Latent Interpolation

## Configuration

- CVAE checkpoint: `checkpoints/fashion_cvae.pth`
- Interpolation steps per example: **8**
- Interpolation rule: `z(alpha) = (1 - alpha) * z_start + alpha * z_end`
- Class conditioning: start label for the first half, end label for the second half

## Examples

| Start class | End class | Seed | Steps |
|---|---|---:|---:|
| Sneaker | Ankle boot | 42 | 8 |
| T-shirt/top | Pullover | 43 | 8 |
| Bag | Sneaker | 44 | 8 |

## Interpretation

The grid demonstrates that nearby points along a straight path in the learned latent space decode into visually related Fashion-MNIST images. Gradual shape and intensity changes indicate that the CVAE learned a structured, continuous representation rather than simply memorizing isolated training examples.

Because FashionGen uses discrete class embeddings, the conditioning label switches at the midpoint of each row. The latent vector itself changes continuously, but the midpoint may show a sharper semantic transition. This is a low-risk exploration feature, not evidence of fully continuous label interpolation.

For the report or presentation, use the grid to explain latent-space continuity, class conditioning, and the distinction between smooth latent movement and discrete semantic control.
