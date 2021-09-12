# rakuten_assignment
My submission to the coding test from Rakuten Paris 

## Prerequisists

Clone the repo and start by install the required packages by running 
```bash
pip install -r requirements.txt
```

## To be noted 

There are more images in the image folder than there are in the csv file, so I had to clean it up. The clean up process, as well as the train, val, split process is all done before the training. 
So in order for the scripts to run *train.py* must be run first. A pretrained model is also provided.

## Launch a training 

Run a training using 
```bash
 python train.py. 
 ```
Use *--max_epoch 2* to get a quick result and be able to run the rest of the commands

* --data_dir : Data containing raw images  
* --csv_file : Data containing csv file 
* --batch_size : Set the size of the batch for the training
* --num_workers : Set the number of workers for the cpu task
* --seed : Set seed for the RNG 
* --learning_rate : Set the learning rate for the Adam optimizer
* --weight_decay : Set the weight decay for the Adam optimizer
* --resume : Resume training from intermediate checkpoint but you need to set the checkpoint 
* --checkpoint : Checkpoint from which to resume 
* --save_ckpt_freq : Frequency at which to save the checkpoint 
* -- tensorboard_display : Enable tensorboard logging

Run in another terminal 
```bash
tensorbord --logdirs runs 
```
to follow the training process. 


## Evaluate the trained model 

Run *python evaluate.py* to evaluate the last model trained. You can choose which one you want to evaluate using the *--checkpoint* command. The metric is the accuracy. 

* --checkpoint : Checkpoint to evaluate. Default : final model from training. A pretrained model is provided. 
* --num_workers : Set the number of workers for the cpu task
* --seed : Set seed for the RNG 
* --test_file': Test csv file 

## Run the flask server

Run *python server.py* to run an Http API to evaluate your own images. 

* --checkpoint  : Pretrained model to use for the inference in the API. The final model from the last training is used by default. 
* --seed : Seed for the RNG

