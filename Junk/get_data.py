from __future__ import print_function
import os
import sys
import gzip
import requests
from six.moves.urllib.request import urlretrieve
from download_helper.MNIST_create import download_and_create_data
from download_helper.SVHN_multi import prep_svhn_multi
from download_helper.SVHN_multi_box import prep_svhn_multi_box

SVHN_new_data_struct_id = '0B4jlyZGFzRIJampwNlk1MjZNTzA'
SVHN_new_data_struct_name = 'SVHN_new_data_struct.pickle'

MNIST_multi_id = '0B4jlyZGFzRIJZlkzXzFmZnVpUVE'
MNIST_multi_name = 'MNIST_multi.hdf5'

SVHN_multi_box_id = '0B4jlyZGFzRIJOHRRMy1ISjVxbEE'
SVHN_multi_box_name = 'SVHN_multi_box.hdf5'

SVHN_multi_id = '0B4jlyZGFzRIJRUQwMWJfN3laQ3M'
SVHN_multi_name = 'SVHN_multi.hdf5'

pickle_location = 'datasets/pickles/'
last_percent_reported = None

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    if not os.path.exists(destination):
        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination) 
        print ('Download of ' + destination + ' Completed.')
        
    else:
        print(destination + ' already present.')

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# Will use 2nd Method if additional command line arguments are given
if (len(sys.argv) > 1):
    maybe_download(pickle_location + 'MNIST_multi.hdf5')

print()
print('You have two Options to get All the Data : ')
print('1. Download the Data from Yann LeCun and SVHN Websites and create Formatted Data Now \n   It can be very Slow.')
print()
print('2. Download pre-processed files prepared by me. \n   It will save you a lot of time. \n   RECOMMENDED')
print()

user_input = input("Please choose method to get Data. \ni.e. Write 1 or 2 depending on your choice    : ")
if user_input == '1' :
    print('   ')
    # print('SVHN_new_data_struct.pickle will be automatically formed during the process')
    print('   ')
    print('------------Processing MNIST_multi.hdf5------------')
    print('---------------------------------------------------')
    download_and_create_data()
    print('-------------MNIST_multi.hdf5 Complete-------------')
    print('   ')
    print('   ')
    # print('----------Processing SVHN_multi_box.hdf5-----------')
    # print('---------------------------------------------------')
    # prep_svhn_multi_box()
    # print('----------SVHN_multi_box.hdf5 Completed------------')
    # print('   ')
    # print('   ')
    print('------------Processing SVHN_multi.hdf5-------------')
    print('---------------------------------------------------')
    prep_svhn_multi()
    print('------------SVHN_multi.hdf5 Completed--------------')
    print('   ')
    print('   ')
    print('All Processes Completed')
    
else :
    print('   ')
    print('Following Files will be Downloaded : \n1. SVHN_new_data_struct.pickle (~80 MB)\n2. MNIST_multi.hdf5 (~2 GB)')
    print('3. SVHN_multi_box.hdf5 (~3 GB)')
    print('Download Location : datasets/pickles/')
    print('   ')
    input('Press Enter to Continue Downloading ~5 GB Data')
    print('   ')
    print('Downloading SVHN_new_data_struct.pickle')
    download_file_from_google_drive(SVHN_new_data_struct_id, pickle_location + SVHN_new_data_struct_name)
    print('   ')
    print('Downloading MNIST_multi.hdf5')
    download_file_from_google_drive(MNIST_multi_id, pickle_location + MNIST_multi_name)
    print('   ')
    print('Downloading SVHN_multi_box.hdf5')
    download_file_from_google_drive(SVHN_multi_box_id, pickle_location + SVHN_multi_box_name)
    print('   ')
    # print('Downloading SVHN_multi.hdf5')
    # download_file_from_google_drive(SVHN_multi_id, pickle_location + SVHN_multi_name)
    # print('   ')
    print('All Downloads Completed.')
    print('------------------------')
