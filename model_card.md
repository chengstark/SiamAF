# Model card for "SiamAF"

Jump to section:

- [Model details](#model-details)
- [Intended use](#intended-use)
- [How to use](#how-to-use)
- [Training dataset](#training-dataset)
- [Training procedure](#training-procedure)
- [Evaluation dataset](#evaluation-dataset)
- [Evaluation results](#evaluation-results)
- [Limitation and bias](#limitations-and-bias)


## Model details

![alt text](https://github.com/chengstark/SiamAF/blob/870a1e763aa7f84ae76db0b57e79672fb6b5fadf/ppgecgjoinarch.png)

SiamAF is a siamese network, designed to learn shared information from both ECG and PPG signals for the  atrial fibrillation (AF) detection. The model is trained on the Instituion A dataset, which contains automatically labeled time-synchronized ECG and PPG signals from bedside monitors. The model has two learning targets $\mathcal{L}_ {\textit{agree}}$ (unsupervised) and $\mathcal{L}_ {CE}$ (Cross-Entropy loss, supervised). Here $\mathcal{L}_ {\textit{agree}}$ defined as follwing:

$$\mathcal{L}_{\textit{agree}}(x^{PPG}_i, x^{ECG}_i) =  - \frac{\left\langle q(z^{ECG}_i), z^{PPG}_i\right\rangle}{\left\|q(z^{ECG}_i)\right\|_{2} \cdot\left\|z^{PPG}_i\right\|_{2}} - \frac{\left\langle q(z^{PPG}_i), z^{ECG}_i\right\rangle}{\left\|q(z^{PPG}_i)\right\|_{2} \cdot\left\|z^{ECG}_i\right\|_{2}}.$$

The complete loss function is defined as follwing with a hyper-parameter $\lambda$:

$$\mathcal{L}_{\textit{joint}}(x^{PPG}_i, x^{ECG}_i, y_i) = \mathcal{L}_{\textit{agree}}(x^{PPG}_i, x^{ECG}_i)
            + \lambda \cdot (\mathcal{L}_{CE}(p^{PPG}_i, y_i)+\mathcal{L}_{CE} (p^{ECG}_i, y_i)).$$

The design of the loss function ecnourages model to learn a similar latent feature for both ECG an PPG signals to extract their shared information and signal characteristics for AF detection (e.g. pulse, peaks, etc.) 

The model is trained using the SGD optimizer. The learning rate is set to 0.1 and momentum set to 0.9. For our proposed method the $\lambda$ is set to 1.

## Intended use

The model is designed for the AF detection task. It is compatiable with either ECG or PPG signals for AF (clas label 1)/non-AF (clas label 0) binary prediction. The input (ECG or PPG) signal should contain 2400 times steps (_length=2400_), ~ 30 seconds.

## How to use

To make prediction on new data samples, you need to install the PyTorch package and numpy package. You will also need the resnet1d.py and dataset.py. Here we provide a sample script loading the trained model weights and makeing prediction on an example dataset in npy format:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F 
from resnet1d import Res34SimSiam
from dataset import Dataset_ori
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict

def test_epoch(MODEL_PATH, test_loader):
    
    state_dict = torch.load(MODEL_PATH) 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] #remove 'module'
        new_state_dict[name] = v

    state_dict = new_state_dict
    model = Res34SimSiam(512, 128, single_source_mode=True).cuda()
    model.load_state_dict(state_dict)

    with torch.no_grad():

        signal_preds = []
        signal_pred_probs = None
        all_targets = None
        
        model.eval()

        for batch_idx, (signal, _) in enumerate(test_loader):

            signal = signal.cuda().float()
            
            _, _, signal_out = model(signal)
            
            signal_predicted = signal_out.argmax(1)
            signal_preds.append(signal_predicted.detach().cpu().numpy())
            
    return np.concatenate(signal_preds)

>>> x_path = '~/x.npy'
>>> y_path = '~/y.npy'
>>> data_x = np.load(x_path)
>>> data_y = np.load(y_path)
>>> print(data_x.shape, data_y.shape)
(100, 2400) (100,)

>>> MODEL_PATH = '~/model.pt'
>>> dataset = Dataset_ori(x_path, y_path)
>>> dataloader = DataLoader(test_dataset, batch_size=2500, shuffle=False)
>>> predictions = test_epoch(MODEL_PATH, testloader)
>>> print(predictions.shape)
>>> print(predictions)
(100,)
[1 1 1 1 1 1 1 1 0 0 0 0 0...1 0]
```

## Training dataset

_The following description is reproduced from the manuscript. Institution A dataset is a private dataset._

The Institution A dataset contains 28539 patients in hospital settings; the patients' continuous ECG and PPG signals were recorded from the bedside monitors. The bedside monitor produced alarms for events including atrial fibrillation (AF), premature ventricular contraction (PVC), Tachy, Couplet, etc. This study focuses on AF, PVC, and normal sinus rhythm (NSR). The samples with PVC and NSR labels were combined into the Non-AF samples group, thus forming the AF vs Non-AF binary classification task. The continuous ECG and PPG signals were sliced into 30-second non-overlap segments (each containing 7,200 timesteps). The 30-second segments were then down-sampled to 2,400 timesteps. During the pre-processing step, invalid samples (e.g. empty signals files, missing ECG or PPG signals) were also filtered out. There are four ECG channels in this dataset, we used the first ECG channel for our study due to its resemblance to wearable device outputs. The dataset is split into the train and validation splits by patient ids. The train split of the Institution A dataset contains 13,432 patients, 2,757,888 AF signal segments, and 3,014,334 Non-AF signal segments; the validation split contains 6,616 patients, 1,280,775 AF segments, and 1,505,119 Non-AF segments. Due to the automatic nature of bedside monitor-generated labels, the dataset likely contains label noise.

## Training procedure

### Preprocessing

The ECG and PPG signals in the training dataset and all test datasets were preprocessed using the following script:

```python
ECG = butter_bandpass_filter(ECG, 0.67, 40, 240, order=1)

PPG = resample(PPG, 2400)
ECG = resample(ECG, 2400)

PPG = (PPG - np.min(PPG)) / (np.max(PPG) - np.min(PPG))
ECG = (ECG - np.min(ECG)) / (np.max(ECG) - np.min(ECG))
```

Each ECG signal strip goes through a band pass filter using the above parameters. Alll ECG and PPG signal subsequently are resampled to 2400 timesteps and normalized into 0-1 range.


## Evaluation dataset

_The following description is reproduced from the manuscript. Institution B dataset is a private dataset._

### Institution B dataset

The Institution B dataset contains 126 patients in hospital settings, and simultaneous continuous ECG and PPG signals were collected at Institution B. The patients have a minimum age of 18 and a maximum age of 95 years old and were admitted from April 2010 to March 2013. The continuous signals were sliced into 30-second non-overlapping segments and downsampled to 2,400 timesteps. The dataset contains 38,910 AF and 220,740 Non-AF segments. A board-certified cardiac electrophysiologist annotated all AF episodes in the Institution B datasets.

### Simband dataset

The Simband dataset contains 98 patients in ambulatory settings from Emory University Hospital (EUH), Emory University Hospital Midtown (EUHM), and Grady Memorial Hospital (GMH). The patients have a minimum age of 18 years old and a maximum age of 89 years old; patients were admitted from October 2015 to March 2016. The ECG signals were collected using Holter monitors, and the PPG signals were collected from a wrist-worn Samsung Simband. The signals used for testing were 30-second segments with 2,400 timesteps after pre-processing. This dataset contains 348 AF segments and 506 Non-AF segments.

### Stanford dataset 

The Stanford dataset contains 107 AF patients, 15 paroxysmal AF patients, and 42 healthy patients. The 42 healthy patients also undergo an exercise stress test. All signals in this dataset were recorded in ambulatory settings. The ECG signals were collected from an ECG reference device, and the PPG signals were collected from a wrist-worn device. The signals were sliced into 25-second segments by the original author. In this study, the signals were also downsampled to 2,400 timesteps. The dataset contains 52,911 AF segments and 80,620 Non-AF segments. In the evaluations, we use the test split generated by the authors of the Stanford dataset.

<!-- ## Evaluation results

This model is evalauted with both AUROC and AUPRC scores with 95% confidence interval, it achieved the following perfromance on test sets:

| Simband (ECG)  | Simband (PPG) | Institution B (ECG) | Institution B (PPG) | Stanford test split |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 0.747 [0.744 0.75]  | 0.914 [0.913 0.916]  | 0.927 [0.925 0.929] | 0.924 [0.922 0.925] | 0.877 [0.876 0.878]| -->


## Limitations and bias

The training labels are auto generated and may contain noise, there is no other inherent limitation or bias to this model.
