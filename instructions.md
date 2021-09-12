# Rakuten France Catalog code test

Hello and welcome to this code test !

## Dataset

You should have received a data file: data.zip
This dataset is composed of product images and their associated category (product type code).
The images are present in the images folder, and the `product type codes` are present in the data_set.csv file

## Task

The goal of this test is to provide an http API to a simple mobilenet V2 classifier.
Write a simple code for training a mobilenet V2 to classify images into product categories.

Three scripts are expected:
- `train.py`: fits the model to data using an adequate loss. 
- `evaluate.py`: test the model according to an appropriate metric (for example: accuracy)
- `server.py`: runs a flask server to which a POST request containing the image can be sent and which (the server) returns the  `product_type_code` for the image.

## Tools and advice

The code has to be written in Python 3 and provided with a requirements.txt file for us to install the environment.
We recommend using PyTorch, but it is also ok to use any other deep learning library.
Keep the code simple and do not hesitate to use Google.
Do not worry about obtaining the best possible performance as we know training hardware can be a limitation.

## Criteria

For the test assessment, we will focus on the following aspects:

- Code quality (e.g., readability, documentation, style).
- Correctness, whether we can successfully run the code and reproduce the expected behaviour.  
- The ability of the candidate to go beyond expectations. Please feel free to impress us.  

## Deliverable

Please send us your code in a zip file or a link to a git repository.
