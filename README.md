goban-image-reader
====================
A tool for transforming goban image to sgf file using deep learning.

This project is still under development. As of August 2018,  the accuracy for a single position (i.e. line intersection) is about 99.8% and  the accuracy for the whole board is about 72%. 

Install
---
1. cd goban-image-reader
2. (optional) Create a virtualenv and activate it. [virtualenv](https://virtualenv.pypa.io/en/stable/userguide/#usage)
3. pip install -r requirements.txt

Usage
---
>to predict images

    python main.py -p xxx/xxx.png yyy/yyy.png zzz/zzz.png


>to create a dataset for evaluation 

    python data_convert.py -d real_test --real_test_image_dir your-image-dir --real_test_sgf_dir your-sgf-dir


>to evaluate a test dataset

    python main.py -er

TODOs
---
* Get more training datas 
* Try using AutoML or similar tools to improve the network
* Add a mechanism for recognizing non-board image
* Enhance the synthetic image to make it more "real"
* Upload to pip (and how?)
