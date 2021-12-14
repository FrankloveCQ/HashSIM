# A pytorch implementation for paper HashSIM

## REQUIREMENTS
1. pytorch 1.8
2. loguru
3. scipy
4. sklearn

## DATASETS
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

## USAGE
```
Hashsim_Pytorch

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset name.
  -r ROOT, --root ROOT  Path of dataset
  -c CODE_LENGTH, --code-length CODE_LENGTH
                        Binary hash code length.(default: 12)
  -T MAX_ITER, --max-iter MAX_ITER
                        Number of iterations.(default: 50)
  -l LR, --lr LR        Learning rate.(default: 1e-3)
  -q NUM_QUERY, --num-query NUM_QUERY
                        Number of query data points.(default: 10000)
  -t NUM_TRAIN, --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 16)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.(default: 24)
  -a ARCH, --arch ARCH  CNN architecture.(default: vgg16)
  -k TOPK, --topk TOPK  Calculate map of top k.(default: -1)
  -v, --verbose         Print log.
  --train               Training mode.
  --evaluate            Evaluate mode.
  -g GPU, --gpu GPU     Using gpu.(default: False)
  -e EVALUATE_INTERVAL, --evaluate-interval EVALUATE_INTERVAL
                        Interval of evaluation.(default: 500)

  ```



## EXPERIMENTS
cifar10: 10000 query images, 5000 training images. Return MAP@ALL(50000) The same setting with DistillHash (19' CVPR )

How to train: python run.py --train 

How to evalutate: python run.py --evaluate

Our model will be restored in the checkpoints.
