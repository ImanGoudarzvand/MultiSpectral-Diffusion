# Evaluations

To compare different generative models, we use FID, sFID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples, which we store in `.npz` (numpy) files.

# Download batches

We provide pre-computed sample batches for the reference datasets, our diffusion models, and several baselines we compare against. These are all stored in `.npz` format.

Reference dataset batches contain pre-computed statistics over the whole dataset, as well as 10,000 images for computing Precision and Recall. All other batches contain 50,000 images which can be used to compute statistics and Precision/Recall.

Here is the link to download all of the sample and reference batches:

STL-10 (reference batch): [google drive](https://drive.google.com/file/d/15pQV8pkuna3TV6CTxqU2-M_KIcROCQ8s/view?usp=sharing)

STL-10 (samples): [google drive](https://drive.google.com/file/d/1fu81uk-jNb_DjTfTWrZBgSqTmWRFF7NM/view?usp=sharing)



# Run evaluations

First, generate or download a batch of samples and download the reference batch for the given dataset. For this example, we'll use STL-10 64x64, so the refernce batch is `stl10_train_reference.npz` and we can use the sample batch `all_sample_tensor_750k_model_wavelet.npz`.

Next, run the `evaluator.py` script. The requirements of this script can be found in [requirements.txt](requirements.txt). Pass two arguments to the script: the reference batch and the sample batch. The script will download the InceptionV3 model used for evaluations into the current working directory (if it is not already present). This file is roughly 100MB.

The output of the script will look something like this, where the first `...` is a bunch of verbose TensorFlow logging:

```
$ python evaluator.py stl10_train_reference.npz all_sample_tensor_750k_model_wavelet.npz
...
computing/reading reference batch statistics...
computing sample batch activations...
100% 782/782 [04:00<00:00,  3.25it/s]
computing/reading sample batch statistics...
Computing evaluations...
Inception Score: 13.872241020202637
FID: 12.553386876600541
sFID: 30.611422134870395
Precision: 0.90664
Recall: 0.666
```
