# Pytorch Image Classification

As of now, it supports only MNIST dataset is used for training and evaluation.

## Training and evaluationg

To run training:
```
python img_classification.py --train
```

To train and evaluate:

```
python img_classification.py --train --eval
```

To evaluate a saved model:

```
python img_classification.py --model PATH_TO_MODEL --eval
```

## Available parameters

```
  --dataset DATASET             Only MNIST option is available now
  --model MODEL                 If you want to evaluate already trained model, pass a
                                path to the model here
  --batch_size BATCH_SIZE       Batch size for training and test
  --output_dir OUTPUT_DIR       Where to save the model
  --epochs EPOCHS               Number of epochs
  --train                       If you need to train
  --eval                        If you need to evaluate
  --learning_rate LEARNING_RATE Learning rate
  --seed SEED                   Set a seed

```

By default the parameters are set as follows:

```
  --dataset         "MNIST"
  --model           None
  --batch_size      64 
  --output_dir      "./output"
  --epochs          15
  --learning_rate   0.003
  --seed            2020
```

## Some Experiments

In the following example the model is trained for 15 epochs using batch size 8. The other parameters are default.

```
python img_classification.py --batch_size 8 --epochs 15 --train --eval
```

Traing with this parameters gives the following results:

```
Test accuracy: 0.8809
```
