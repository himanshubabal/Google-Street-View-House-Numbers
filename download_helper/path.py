import os


gcloud_dir = '/home/himanshubabal/Google-Street-View-House-Numbers/datasets/'
external_hdd = '/Volumes/700_GB/Study/SVHN/SVHN-Full_Dataset/'
mac_dir = '/Users/himanshubabal/Google Drive/Himanshu - 1 Drive/ML/Assignments/Udacity/ML Nanodegree Projects/P5 Digit Recognition/datasets/'

if os.path.exists(gcloud_dir):
    data_dir = gcloud_dir

elif os.path.exists(external_hdd):
    data_dir = external_dir

else:
    data_dir = mac_dir

# Google Cloud
# data_dir = '/home/himanshubabal/Google-Street-View-House-Numbers/datasets/'

# Mac local
# data_dir = '/Users/himanshubabal/Google Drive/Himanshu - 1 Drive/ML/Assignments/Udacity/ML Nanodegree Projects/P5 Digit Recognition/datasets/'

# External Hard Drive
# data_dir = '/Volumes/700_GB/Study/SVHN/SVHN-Full_Dataset/'
