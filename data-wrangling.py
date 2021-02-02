# Data wrangling for DAPS, RIRs, and Noise datsets for HiFi-GAN

import os
import sys
import shutil
# TODO unzip DAPS, RIRs, and Noise datasets
# from zipfile import ZipFile
# with ZipFile('rirs_noises.zip', 'r') as zipobj:
#   zipobj.extractall()

# filenames = os.listdir('.')
# for f in filenames:
#   if f != 'daps_dataset.zip':
#     with ZipFile(f, 'r') as zipobj:
#       zipobj.extractall(path='../DAPS')
# from zipfile import ZipFile
# with ZipFile('Audio.zip', 'r') as zipobj:
#   zipobj.extractall()
# from zipfile import ZipFile
# with ZipFile('daps_dataset.zip', 'r') as zipobj:
#   zipobj.extractall()

def dir_walk(path, ext):
    file_list = [os.path.join(root, name)
                 for root, dirs, files in os.walk(path)
                 for name in sorted(files)
                 if name.endswith(ext)
                 and not name.startswith('.')]
    return file_list

def make_paths(path, subpaths=[]):
  if subpaths:
    for subpath in subpaths:
      newpath = path + '/' + subpath
      if not os.path.exists(newpath):
        os.makedirs(newpath)
      return newpath
  else:
    if not os.path.exists(path):
      os.makedirs(path)
  return path


# TODO train test split
# create train and test folders in repo folder
# TODO take paths from cli argument
# train_path = ''
# test_path = ''


# filenames = dir_walk(os.getcwd(), ext=('.wav', '.WAV'))
# file_ids = os.listdir('.')

# for i in range(len(filenames)):
#   # move first 200 AIRs to train/RIRS
#   if i <=200:
#     shutil.copyfile(filenames[i], train_path+'/IRs/'+file_ids[i])
#   # move everything else to test/RIRS
#   else:
#     shutil.copyfile(filenames[i], test_path+'/IRs/'+file_ids[i])

# take location of Datasets/DAPS as cli arg
source = str(sys.argv[1])
destination = make_paths(str(sys.argv[2]))
filenames = dir_walk(source, ext=('.wav', '.WAV'))
file_ids = [name for root, dir, files in os.walk(source) for name in sorted(files)]
# For each iphone/ipad folder, take everything with "script5"
for i in range(len(filenames)):
  if ("script5" in filenames[i] and "iphone" in filenames[i]) or ("script5" in filenames[i] and "ipad" in filenames[i]):
    print(filenames[i])
    shutil.copyfile(filenames[i], destination + '/' + file_ids[i])
    print("moved " + file_ids[i])