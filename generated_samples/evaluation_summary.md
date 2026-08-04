# FashionGen Studio Evaluation Summary

## Run configuration

- Seed: `42`
- Total generated images: **80**
- CVAE checkpoint: `checkpoints/fashion_cvae.pth`
- Classifier checkpoint: `checkpoints/fashion_classifier.pth`

## Overall results

- Passed quality checks: **66/80**
- Overall pass rate: **82.5%**
- Average classifier confidence: **0.878**
- Classifier checkpoint test accuracy: **90.61%**
- CVAE checkpoint validation loss: **57.6404**

## Per-class quality-check results

| Class index | Requested class | Passed | Total | Pass rate | Average confidence |
|---:|---|---:|---:|---:|---:|
| 0 | T-shirt/top | 7 | 8 | 87.5% | 0.971 |
| 1 | Trouser | 8 | 8 | 100.0% | 1.000 |
| 2 | Pullover | 7 | 8 | 87.5% | 0.896 |
| 3 | Dress | 8 | 8 | 100.0% | 0.908 |
| 4 | Coat | 5 | 8 | 62.5% | 0.750 |
| 5 | Sandal | 6 | 8 | 75.0% | 0.818 |
| 6 | Shirt | 3 | 8 | 37.5% | 0.566 |
| 7 | Sneaker | 6 | 8 | 75.0% | 0.899 |
| 8 | Bag | 8 | 8 | 100.0% | 0.975 |
| 9 | Ankle boot | 8 | 8 | 100.0% | 0.999 |

## Interpretation for report and slides

The CNN agreed with the requested class for 82.5% of CVAE samples, with average confidence 0.878. This agreement rate is a practical automated quality proxy: it measures whether generated class features are recognizable to an independently trained Fashion-MNIST classifier.

The strongest row was **Ankle boot** at 100.0%. The weakest class(es) were **Shirt** at 37.5%. Lower-performing classes likely reflect visual overlap among Fashion-MNIST clothing categories or less distinct CVAE samples and should be highlighted as improvement areas.

The classifier agreement score is not a substitute for human visual assessment. The accompanying grid should be used to discuss image clarity, diversity, and recognizable class structure in the final report and presentation.
