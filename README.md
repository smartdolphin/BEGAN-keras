# BEGAN-keras
A Keras implementation of the [Boundary Equilibrium GAN Paper](https://arxiv.org/abs/1703.10717)

## Extract faces data
Please extract image data in the faces folder.<br>

    $ unzip image_data.zip

Input data

    └── began
        └── faces
            └── image_data
                └── xxx.png (name doesn't matter)


## Usage
Install the necessary requirements. Import your data into the main.py file and make the appropriate parameter changes. The correctness of this implementation is still being verified.<br>


<b>Training locally</b>

    $ python main.py --batches_per_epoch=50 --batch_size=10 --image_dir=./out
    

<b>Training in Cloud</b>
```
BUCKET_NAME=gs://{user}
TRAIN_DATA=gs://{user}/train_data
IMAGE_DIR=gs://{user}/out
JOB_NAME=began_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=began --module-name=began.main \
--staging-bucket=$BUCKET_NAME --region=asia-east1 \
--config=began/cloudml-gpu.yaml \
-- --train_data=$TRAIN_DATA --image_dir=$IMAGE_DIR
```

## Requirements
- Python 2.7 or 3
- Keras
- Numpy
- Matplotlib
- Tensorflow or Theano
