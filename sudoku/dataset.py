###################################################################################
# The Chars74K dataset
# Character Recognition in Natural Images
# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

# EnglishFnt.tgz (51.1 MB): characters from computer fonts with 4 variations 
# (combinations of italic, bold and normal). Kannada (657+ classes) 
# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz
###################################################################################.

import os
import pickle
import numpy as np
from sys import exit
from cv2 import resize
from shutil import move, rmtree
from matplotlib import pyplot as plt

def dataset(filename='digits-dataset'):
  """
  Process and dump to pickle file the The Chars74K dataset.
  PS.: It only get the first 10 folders --> equivalent to digits [0-9] <--
  """

  root_folder = './EnglishFnt/English/Fnt/'
  new_root_folder = './dataset/digits/'

  if not os.path.exists(root_folder):
    print('Folder not found. To continue, please download and extract the \
          file from www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz')
    exit(0)

  # else:
  #   print('Folder found.')

  if not os.path.exists(new_root_folder):
    os.makedirs(new_root_folder)
    print('Folder {} created'.format(new_root_folder))

  # Get only the folders that contains numbers
  sub_folders = sorted(os.listdir(root_folder))[:10]

  # Copy sub_folder content to new location
  for folder in sub_folders:
    move(os.path.join(root_folder, folder), os.path.join(new_root_folder, folder))
    print('Folders moved from {} to {}'.format(root_folder, new_root_folder))

  # Updating sub_folders
  sub_folders = sorted(os.listdir(new_root_folder))

  # read image files
  digits = {}
  for i, folder in enumerate(sub_folders):
    digits[i] = [(resize(1 - plt.imread(os.path.join(new_root_folder, folder, img)), (28, 28))).reshape(-1) for 
                img in os.listdir(os.path.join(new_root_folder, folder))]  
    digits[i] = np.asarray(digits[i])
    
  # sample training and test sets
  num_classes = 10

  for i in digits:
    np.random.shuffle(digits[i])
    label_row = np.zeros(num_classes)
    label_row[i] = 1
    label_row = label_row.reshape(1, -1)

    labels = np.repeat(label_row, len(digits[i]), axis=0)
      
    if i == 0:
      x_train = digits[i][:900]
      y_train = labels[:900]
      x_test = digits[i][900:]
      y_test = labels[900:]
    else: 
      x_train = np.vstack([x_train, digits[i][:900]])
      y_train = np.vstack([y_train, labels[:900]])
      x_test = np.vstack([x_test, digits[i][900:]])
      y_test = np.vstack([y_test, labels[900:]]) 

  # Dump files
  dataset_folder = './dataset/'
  if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

  file_dir = os.path.join(dataset_folder, filename)

  with open(file_dir, 'wb') as f:
    pickle.dump([x_train, y_train, x_test, y_test], f)
    print('Variables dumped to {}'.format(file_dir))

  # Deleting folders:
  try:
    rmtree('./EnglishFnt/')
    rmtree('./dataset/digits/')
    os.remove('EnglishFnt.tgz')
    print('Folder deleted')
  except:
    pass

