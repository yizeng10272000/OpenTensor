# OpenTensor

This repository is adapted from the original [OpenTensor](https://github.com/YiwenAI/OpenTensor) repository by YiwenAI.
The modified version is maintained here: [https://github.com/yizeng10272000/OpenTensor/tree/OpenTensor](https://github.com/yizeng10272000/OpenTensor/tree/OpenTensor).

This repository is adapted from the original implementation of the paper "OpenTensor: Reproducing Faster Matrix Multiplication Discovering Algorithms" presented at the 37th Conference on Neural Information Processing Systems Workshop (NeurIPS 2023) by Yiwen Sun and Wenye Li.

We provide the code to generate synthetic tensors, train OpenTensor, and perform tensor decomposition.

---

## Config

All configurations should be contained in a YAML file. We provide some config templates in the `./config` folder. For example, `./config/S_4.yaml` is the config file for decomposing a $4 \times 4 \times 4$ matrix multiplication tensor, which is equivalent to discovering the $2 \times 2$ matrix multiplication algorithm.

---

## 1. Generate synthetic data

```bash
mkdir data
mkdir exp
python main.py --config ./config/S_4.yaml --mode generate_data
```

- The generated data will be stored in the folder `/data`.
- `S_4.yaml` is the config for the training process.
- This command generates 100,000 synthetic tensors and saves them in the `./data` folder.

---

## 2. Train and compare two models (original and new with tensor projection) simultaneously

```bash
python compare.py
```

- The trained models will be saved in the folder `/exp`.
- The comparison result will be saved as `compare_result.csv`.

---

## 3. Train and test a single model

### 3.1 Train and test the model **with tensor projection**

**Train the model:**

```bash
python main.py --config ./config/S_4_with_projection.yaml --mode train
```

- The trained model will be saved in `/exp/with_projection/<timestamp>/ckpt/latest.pth`.
- `latest.pth` is the trained model checkpoint.
- `<timestamp>` is a random timestamp generated for each training process.

**Test the trained model:**

```bash
python main.py --config ./config/S_4_with_projection.yaml --mode infer --run_dir ./exp/with_projection/<timestamp>/ckpt/latest.pth
```

- The test results will be saved in `/exp/with_projection/<timestamp>/infer/<timestamp>.txt`.

---

### 3.2 Train and test the model **without tensor projection**

**Train the model:**

```bash
python main.py --config ./config/S_4_without_projection.yaml --mode train
```

- The trained model will be saved in `/exp/without_projection/<timestamp>/ckpt/latest.pth`.
- `latest.pth` is the trained model checkpoint.
- `<timestamp>` is a random timestamp generated for each training process.

**Test the trained model:**

```bash
python main.py --config ./config/S_4_without_projection.yaml --mode infer --run_dir ./exp/without_projection/<timestamp>/ckpt/latest.pth
```

- The test results will be saved in `/exp/without_projection/<timestamp>/infer/<timestamp>.txt`.

---

## Citing us

If our work has been helpful to you, please feel free to cite us:

```latex
@article{sun2024opentensor,
  title={OpenTensor: Reproducing Faster Matrix Multiplication Discovering Algorithms},
  author={Sun, Yiwen and Li, Wenye},
  journal={arXiv preprint arXiv:2405.20748},
  year={2024}
}
```

---

Thank you for using this project! Please feel free to open issues or contact us if you have any questions.
