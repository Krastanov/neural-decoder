# Neural Network Decoders for Quantum Error Correcting Codes

See: https://www.nature.com/articles/s41598-017-11266-1

Make your own decoder with:

```
train_network.py 5 output.model \
  --onthefly 10000000 50000 \
  --Xstab --Zstab \
  --epochs 10 --prob 0.9 \
  --learningrate .000001 --normcenterstab --layers 4 4 4 4 4 4 4
```
Test a network by adding the `--eval` flag.

See `train_network.py -h` for description of each flag.
