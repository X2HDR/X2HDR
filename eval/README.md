# RAW-to-HDR Evaluation

We use [HDRQA](https://github.com/cpb68/HDRQA) for RAW-to-HDR reconstruction evaluation. Our implementation closely follows the original code, with minor modifications.

To evaluate the results:

1. Open `HDRMetrics.py`.
2. Modify lines 77–78 to specify the paths to the source images and ground-truth HDR images.
3. Run the following command:

```shell
python HDRMetrics.py
```

The evaluation metric can be changed on line 210.