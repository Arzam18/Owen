# Trainer

## Requirements

```bash
pip install torch numpy
# GPU: install CUDA build of torch per pytorch.org
```

## Generate sdata

```bash
cmake --build build -j
./build/owen2-sdata --games 200000 --out data/sdata.bin
ls -lh data/sdata.bin
```

Record size is 69 bytes (pragma pack 1). See `sdata.py`.

## Train

```bash
python trainer/train.py --sdata data/sdata.bin --out nets/o2-v1.o2nn --epochs 80 --gpu
# CPU fallback:
python trainer/train.py --sdata data/sdata.bin --out nets/o2-v1.o2nn --epochs 20 --device cpu
```

The trainer exports `.o2nn` (Owen 2 NNUE format). Load in engine:

```bash
./build/owen2
> setoption name NNUEFile value nets/o2-v1.o2nn
> isready
```

## Notes

- Loss blends game result (WDL) and search eval (distillation). Edit `SDataDataset.__getitem__` to change blend.
- `feature_indices` must match `src/nnue/features.cpp`.
- Use `--batch 8192` on large GPUs; lower to 1024 on CPU/MPS.
