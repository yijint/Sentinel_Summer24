########
#
# cxl-snapshot-relabeling.py
#
# Demonstrate the labelme box correction process on a folder of images.
#
# If you are running on Windows, you will need to run this notebook as a user with 
# admin privileges to create symlinks.
#
# Things you might want to do before running this notebook:
#
# * Resize all your images to ~1600px on the long side; I use resize_image_folder for this:
#
#   https://megadetector.readthedocs.io/en/latest/visualization.html#megadetector.visualization.visualization_utils.resize_image_folder
#
#   ...which can resize a folder of images to a new folder on multiple threads/processes.
#
# * Run MegaDetector and do RDE, e.g. with this notebook:
#    
#   https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.py
#
#   ...or if you're working with LILA data, just download the results file after RDE:
#
#   https://lila.science/megadetector-results-for-camera-trap-datasets    
#
#   Either way, this notebook assumes you've already run MD.
#
# Non-standard depdenencies:
#
# pip install megadetector
# pip install clipboard    
# 
########

#%% Imports and constants

import os
import json
import sys

# I ran MegaDetector before running this script; this is the results file.
md_results_file = r'g:\temp\B01\mdv5a.json'

# This is where the images live; it should be the same folder on which you ran MegaDetector
relabeling_folder_base = r'g:\temp\B01'

# Subfolders will be created within this folder; each subfolder will have the symlinks for a particular
# labeling chunk
symlink_folder = r'g:\temp\cxl-relabeling-symlinks'

# Normalize paths
md_results_file = md_results_file.replace('\\','/')
relabeling_folder_base = relabeling_folder_base.replace('\\','/')
symlink_folder = symlink_folder.replace('\\','/')
use_threads = True
n_workers = 10

batch_name = 'snapshot-cxl'
max_images_per_chunk = 5000

# This is a hack I use because sometimes I run this process in WSL instead of native Windows
if sys.platform != 'win32':
    md_results_file = md_results_file.replace('c:/','/mnt/c/')
    relabeling_folder_base = relabeling_folder_base.replace('c:/','/mnt/c/')
    symlink_folder = symlink_folder.replace('c:/','/mnt/c/')
    use_threads = False

assert os.path.isfile(md_results_file)
assert os.path.isdir(relabeling_folder_base)
os.makedirs(symlink_folder,exist_ok=True)

default_confidence_threshold = 0.2

# This defines the set of backup label files we generate from MD results at lower confidence thresholds
index_to_threshold = {
    1:0.1,
    2:0.05,
    3:0.01
}

# I use slightly different conventions for file IDs for difference projects
def get_file_id(fn):
    return os.path.basename(fn).split('.')[0]


#%% Convert MD results to labelme format with a default threshold

from megadetector.postprocessing.md_to_labelme import md_to_labelme

_ = md_to_labelme(results_file=md_results_file,
                  image_base=relabeling_folder_base,
                  confidence_threshold=default_confidence_threshold,
                  overwrite=True,
                  extension_prefix='',
                  n_workers=n_workers,
                  use_threads=use_threads,
                  bypass_image_size_read=False,
                  verbose=True)


#%% Create alternative .json files based on MD results at lower thresholds

for index in index_to_threshold.keys():
    
    print('Generating alternative labels for index {} (threshold {})'.format(
        index,index_to_threshold[index]))
    
    md_to_labelme(results_file=md_results_file,
                  image_base=relabeling_folder_base,
                  confidence_threshold=index_to_threshold[index],
                  overwrite=True,
                  use_threads=use_threads,
                  bypass_image_size_read=False,
                  extension_prefix='.alt-{}'.format(index),
                  n_workers=n_workers)


#%% Enumerate files

from megadetector.utils.path_utils import recursive_file_list

all_files_relative = recursive_file_list(relabeling_folder_base,
                                         return_relative_paths=True,
                                         convert_slashes=True,
                                         recursive=True)

print('Enumerated {} files'.format(len(all_files_relative)))


##%% Match .json files to images

from megadetector.utils.path_utils import find_image_strings
image_files_relative = find_image_strings(all_files_relative)
json_files = [fn for fn in all_files_relative if fn.endswith('.json')]
json_files = sorted(json_files)

print('Enumerated {} image files and {} .json files'.format(
    len(image_files_relative),len(json_files)))


##%% Group json files by the image they belong to

# We'll use this to create symlinks to every file that goes with each image in
# a chunk.

from collections import defaultdict
from tqdm import tqdm

image_file_base_to_json_files = defaultdict(list)

# json_file = json_files[0]
for json_file in tqdm(json_files):

    file_id = get_file_id(json_file)
    image_file_base_to_json_files[file_id].append(json_file)


##%% Make sure every image has the right number of .json files

unlabeled_image_files = []

for image_file in tqdm(image_files_relative):    
    basename = get_file_id(image_file)
    json_files_this_image = image_file_base_to_json_files[basename]
    assert len(json_files_this_image) == 4
    if len(json_files_this_image) == 0:
        unlabeled_image_files.append(image_file)

    
#%% Divide into chunks, create symlinks

from megadetector.utils.ct_utils import split_list_into_fixed_size_chunks
from megadetector.utils.path_utils import safe_create_link

chunks = split_list_into_fixed_size_chunks(image_files_relative,max_images_per_chunk)

print('Split images into {} chunks of {} images'.format(len(chunks),max_images_per_chunk))

chunk_folder_base = os.path.join(relabeling_folder_base,'symlinks-{}'.format(batch_name))
chunk_folders = []
error_files = []

# i_chunk = 0; chunk = chunks[i_chunk]
for i_chunk,chunk in enumerate(chunks):
    
    print('Creating symlinks for chunk {} of {}'.format(i_chunk,len(chunks)))

    chunk_folder_abs = os.path.join(chunk_folder_base,'chunk_{}'.format(
        str(i_chunk).zfill(3)))
    os.makedirs(chunk_folder_abs,exist_ok=True)
    chunk_folders.append(chunk_folder_abs)
    
    # Find matching files
    relative_files_this_chunk = []
    
    # i_image=0; image_file = chunk[i_image]
    for i_image,image_file in enumerate(chunk):
        
        # image_file_abs = os.path.join(training_images_resized_folder,image_file); open_file(image_file_abs)
        basename = get_file_id(image_file)
        json_files_this_image = image_file_base_to_json_files[basename]
        
        # These are typically images that failed to load
        if len(json_files_this_image) == 0:
            print('Warning: no .json files for {}'.format(image_file))
            error_files.append(image_file)
            continue
        
        assert len(json_files_this_image) > 0
        relative_files_this_chunk.append(image_file)
        
        for json_file in json_files_this_image:
            relative_files_this_chunk.append(json_file)            
    
    # Create symlinks
    #
    # relative_file = relative_files_this_chunk[0]
    for relative_file in tqdm(relative_files_this_chunk):
        source_file_abs = os.path.join(relabeling_folder_base,relative_file)
        assert os.path.isfile(source_file_abs)
        target_file_abs = os.path.join(chunk_folder_abs,relative_file)
        os.makedirs(os.path.dirname(target_file_abs),exist_ok=True)
        safe_create_link(source_file_abs,target_file_abs)

# ...for each chunk

error_file_list_file = os.path.join(chunk_folder_base,'error_images.json')
print('\nSaving list of {} error images to {}'.format(len(error_files),error_file_list_file))
with open(error_file_list_file,'w') as f:
    json.dump(error_files,f,indent=1)


#%% Label one chunk

# Specifically, generate the command to start labelme, pointed at this chunk, and copy that
# command to the clipboard.

i_chunk = 0
resume = True

chunk_folder_abs = os.path.join(chunk_folder_base,'chunk_{}'.format(
    str(i_chunk).zfill(3)))
assert os.path.isdir(chunk_folder_abs)

flags = ['ignore','empty']

flag_file = os.path.join(chunk_folder_abs,'flags.txt')
with open(flag_file,'w') as f:
    for flag in flags:        
        f.write(flag + '\n')

last_updated_file = os.path.join(chunk_folder_abs,'labelme_last_updated.txt')
cmd = 'python labelme "{}" --labels animal,person,vehicle --linewidth 12 --last_updated_file "{}" --flags "{}"'.format(
    chunk_folder_abs,last_updated_file,flag_file)
if resume:
    cmd += ' --resume_from_last_update'
import clipboard; print(cmd); clipboard.copy(cmd)
