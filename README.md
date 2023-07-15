# MAADFace-HQ
MAADFAce-HQ is the combination of [MAAD-Face](https://github.com/pterhoer/MAAD-Face) and [VGGFace2-HQ](https://github.com/NNNNAI/VGGFace2-HQ).

## Acquiring dataset
To get MAADFace-HQ dataset, First, get the VGGFace2-HQ dataset from its offical repository. The training set can be found on the link it provided. As for testing set, get the VGGFace2 test set from [kaggle](https://www.kaggle.com/datasets/greatgamedota/vggface2-test), then use the code offer from VGGFace2-HQ repository to convert it into high quaility verision. python 3.6 is recommended.

MAAD-Face repository only provide label for VGGFace2. But the VGGFace2-HQ drop some of the image during converion. So we put the training and test set in their corresponding folders, find out what's remain, then output the final lables.

```
python maadface_hq_split --in-csv MAAD_Face.csv --train-folder ./train --test-folder ./test --train-csv MAADFace_HQ_train.csv --test-csv MAADFace_HQ_test.csv
```

This repository explore the fairness on the model trained on MAADFace-HQ dataset. The sensitive attribute is "gender".

