# Quantile-based Maximum Likelihood Training for Outlier Detection (QuantOD)

Welcome to the official implementation repository for **Quantile-based Maximum Likelihood Training for Outlier Detection**. To dive deeper into our methodology, please consult our [paper](https://arxiv.org/).

**Authors:** Taghikhah*, Kumar*, Segvic, Eslami, and Gumhold.

Our research introduces a quantile-based maximum likelihood objective to learn the inlier distribution, aiming to enhance the separation between inliers and outliers during inference. By adapting a normalizing flow to pre-trained discriminative features, our technique efficiently flags outliers based on their log-likelihood evaluations.

![QuantOD Visualization](https://github.com/taghikhah/QuantOD/blob/main/images/QuantOD.png)

## Instructions

Before diving into experiments, ensure you have the necessary libraries and packages installed. To simplify this process, we have provided a `requirements.txt` file. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Once the installations are complete, you can proceed with the experiments.

### Datasets

Our experiments employ CIFAR-10 and CIFAR-100 as inlier datasets, complemented by several outlier datasets:

- [Textures](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Cimpoi_Describing_Textures_in_2014_CVPR_paper.pdf)
- [Places365](http://places2.csail.mit.edu/PAMI_places.pdf)
- [LSUN-C](https://www.researchgate.net/profile/Jianxiong-Xiao/publication/278048515_LSUN_Construction_of_a_Large-scale_Image_Dataset_using_Deep_Learning_with_Humans_in_the_Loop/links/612e52370360302a006f1f49/LSUN-Construction-of-a-Large-scale-Image-Dataset-using-Deep-Learning-with-Humans-in-the-Loop.pdf)
- [LSUN-R](https://www.researchgate.net/profile/Jianxiong-Xiao/publication/278048515_LSUN_Construction_of_a_Large-scale_Image_Dataset_using_Deep_Learning_with_Humans_in_the_Loop/links/612e52370360302a006f1f49/LSUN-Construction-of-a-Large-scale-Image-Dataset-using-Deep-Learning-with-Humans-in-the-Loop.pdf)
- [iSUN](https://)
- [SVHN](https://)

First-time users will have these datasets automatically downloaded to:

```
./data/
```

### Pretrained Models

Upon the initial script execution, pretrained models will auto-download to:

```
./weights/
```

Note: The provided pretrained QuantOD model utilizes default hyperparameters. Deviating from these parameters necessitates retraining the flows network.

### Testing

Test QuantOD for **CIFAR-10 WideResnet-40-2**:

```bash
python test.py -em QuantOD -id cifar10 -ood all -cm wideresnet40 -cw 'weights/cifar10_wideresnet40_last.pt' -fm glow -fw 'weights/cifar10_glow_last.pt'
```

For **CIFAR-100 WideResnet-40-2**:

```bash
python test.py -em QuantOD -id cifar100 -ood all -cm wideresnet40  -cw 'weights/cifar100_wideresnet40_last.pt' -fm glow -fw 'weights/cifar100_glow_last.pt'
```

### Training

Train QuantOD on **CIFAR-10 WideResnet-40-2**:

```bash
python train.py -id cifar10 -cm wideresnet40 -cw 'weights/cifar10_wideresnet40_last.pt'
```

Train QuantOD on **CIFAR-100 WideResnet-40-2**:

```bash
python train.py -id cifar100 -cm wideresnet40 -cw 'weights/cifar100_wideresnet40_last.pt'
```

---

#### Training Hyperparameters

1. **Classifier Model Selection** (`--classifier-model` or `-cm`):

   - **Description**: Choose the classifier architecture.
   - **Choices**: `wideresnet40`, `mobilenetv2`, `resnet18`, `densenet121`
   - **Default**: `wideresnet40`

   ```bash
   Example: python train.py --classifier-model resnet18
   ```

2. **Quantile NLL Loss Function** (`--q-nll` or `-qn`):

   - **Description**: Adjust the quantile loss function.
   - **Range**: `0.0` to `1.0`
   - **Default**: `0.05`
   - **Values**:
     - `0.0`: Minimum
     - `0.05`: 5-th quantile
     - `0.1`: Decile
     - `0.25`: Quartile
     - `0.5`: Median
     - `1.0`: Mean

   ```bash
   Example: python train.py --q-nll 0.1
   ```

3. **Coupling Blocks Architecture** (`--flows-model` or `-fm`):

   - **Description**: Specify the architecture of coupling blocks.
   - **Choices**: `realnvp`, `glow`, `nice`, `gin`
   - **Default**: `glow`

   ```bash
   Example: python train.py --flows-model realnvp
   ```

4. **Coupling Blocks Count** (`--flows-steps` or `-fs`):

   - **Description**: Set the number of coupling blocks in the flow model.
   - **Default**: `8`

   ```bash
   Example: python train.py --flows-steps 10
   ```

5. **Layers in Coupling Network** (`--flows-layers` or `-fl`):

   - **Description**: Define the number of layers in the coupling network.
   - **Default**: `2`

   ```bash
   Example: python train.py --flows-layers 3
   ```

6. **Neurons in Coupling Network** (`--flows-hidden` or `-fh`):

   - **Description**: Determine the number of neurons for the coupling networks.
   - **Default**: `512`

   ```bash
   Example: python train.py --flows-hidden 256
   ```

7. **Dropout Rate in Coupling Networks** (`--flows-dropout` or `-fd`):
   - **Description**: Specify the dropout rate for the coupling networks.
   - **Default**: `0.3`
   ```bash
   Example: python train.py --flows-dropout 0.5
   ```
