# Atomic Commands

## Atomic Commands for R2M process

If there is a demand that process the `FrcSub` dataset from `rawdata` to `middata`, we can run the following command.
```bash
edustudio r2m R2M_FrcSub --dt FrcSub --rawpath data/FrcSub/rawdata --midpath data/FrcSub/middata
```

The command would read raw data files from `data/FrcSub/rawdata` and then save the middata in `data/FrcSub/middata`

