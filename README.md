# A-code-framework-of-infrared-small-detection 
This is an improved network based on the U-Net architecture.
You'll need to create a new folder called "dataset" to store different datasets. Each dataset should be placed in its own subfolder, and the name of the folder can be chosen at your discretion. Each dataset include images, masks and split method, and the split method include two text files with the names of the training images and the test images. The file structure is as follows:
└─ICPR_Track2
    ├─70_20
    │      full.txt
    │      statistics.txt
    │      test.txt
    │      train.txt
    │      
    ├─images
    │      00001.png
    │      00002.png
    │      ...
    └─masks
            00001.png
            00002.png
            ...
# Python versions and deeplearning frameworks
# python = 3.11
# pytorch = 2.2.1
# cuda = 11.8

Finally, you can install required packages by the following code:
pip install -r requirements.txt
