# Run MDv5a and RDE on a dataset  
# adapted from https://github.com/agentmorris/MegaDetector/blob/main/notebooks/manage_local_batch.ipynb

# import packages
import json
import sys
import os
import stat
import time
import re
from glob import glob
from datetime import datetime
import time
import pandas as pd
import uuid
import argparse
import sqlalchemy
import random

import humanfriendly

from tqdm import tqdm
from collections import defaultdict

from megadetector.visualization.visualization_utils import resize_image_folder 

from megadetector.utils import path_utils
from megadetector.utils.path_utils import find_image_strings
from megadetector.utils.path_utils import recursive_file_list
from megadetector.utils.path_utils import safe_create_link
from megadetector.utils.ct_utils import split_list_into_n_chunks
from megadetector.utils.ct_utils import image_file_to_camera_folder
from megadetector.utils.ct_utils import split_list_into_fixed_size_chunks

from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.detection.run_detector import DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
from megadetector.detection.run_detector import estimate_md_images_per_second
from megadetector.detection.run_detector import get_detector_version_from_filename

from megadetector.postprocessing.jin_md_to_labelme import md_to_labelme
from megadetector.postprocessing.postprocess_batch_results import PostProcessingOptions, process_batch_results
from megadetector.postprocessing.repeat_detection_elimination import repeat_detections_core
from megadetector.postprocessing.repeat_detection_elimination import remove_repeat_detections

def mdv5a_and_rde(input_path, job_name, job_date,
                run_md_automatically=True, n_jobs=1, n_gpus=1, default_gpu_number=0, ncores=1,
                verbose = False):
    '''
    Run MDv5a and RDE on the given dataset at input_path. 

    Args:
        input_path (str): Path to data folder, e.g., '/home/user/data'
        job_name (str): A short name for this job, e.g. 'nz-trailcams'
        job_date (str): Date of running this job, e.g. '2024-jun-07'
        run_md_automatically (bool): Whether to run megadetector automatically; if False, 
                                    scripts have to be run manually from the terminal.
                                    If True, there will be no parallelization over multiple 
                                    processes, so the tasks will run serially. 
                                    This only matters if you have multiple GPUs. 
        n_jobs (int): Number of jobs to split data into, typically equal to the number of 
                    available GPUs though when using augmentation or an image queue (and thus 
                    not using checkpoints), Dan Morris typically uses ~100 jobs per GPU; 
                    those serve as de facto checkpoints.
        n_gpus (int): Number of available GPUs
        default_gpu_number (int): Only relevant for single-GPU usage
        ncores (int): Number of cores available; only relevant when running on CPU
        verbose (bool): Whether to print job details 
    
    NOTE: 
    the user does not have to understand what the outputs are, they can be fed directly as arguments
    to post_rde(), also defined in this script. 

    Return: 
        combined_api_output_file (str): filepath to megedetection results on the whole dataset 
        rde_string (str): string containing rde settings 
        suspicious_detection_results (object): repeat detection results 
        default_workers_for_parallel_tasks (int): settings for image rendering
        parallelization_defaults_to_threads (bool): prefer threads on Windows, processes on Linux
        postprocessing_output_folder (str): folder path to filtered output 
        base_task_name (str): naming conventions 
    '''

    if verbose:
        print("-------------------------------", flush=True)
        print(f"job_name: {job_name}", flush=True)
        print(f"job_date: {job_date}", flush=True)
        print(f"input_path: {input_path}", flush=True)
        print(flush=True)
        print(f"n_jobs: {n_jobs}", flush=True)
        print(f"n_gpus: {n_gpus}", flush=True)
        print(f"default_gpu_number: {default_gpu_number}", flush=True)
        print(f"ncores: {ncores}", flush=True)
        print("-------------------------------", flush=True)

    #################################################################################################
    # SET CONSTANTS                                                                                 #
    #################################################################################################

    ## Inference options
    # To specify a non-default confidence threshold for including detections in the .json file
    json_threshold = None

    # Turn warnings into errors if more than this many images are missing
    max_tolerable_failed_images = 100

    # Should we supply the --image_queue_option to run_detector_batch.py?  I only set this
    # when I have a very slow drive and a comparably fast GPU.  When this is enabled, checkpointing
    # is not supported within a job, so I set n_jobs to a large number (typically 100).
    use_image_queue = False

    # Should we supply --quiet to run_detector_batch.py?
    quiet_mode = True

    # Specify a target image size when running MD... strongly recommended to leave this at "None"
    # When using augmented inference, if you leave this at "None", run_inference_with_yolov5_val.py
    # will use its default size, which is 1280 * 1.3, which is almost always what you want.
    image_size = None

    # Should we include image size, timestamp, and/or EXIF data in MD output?
    include_image_size = False
    include_image_timestamp = False
    include_exif_data = False

    # OS-specific script line continuation character (modified later if we're running on Windows)
    slcc = '\\'

    #  OS-specific script comment character (modified later if we're running on Windows)
    scc = '#'

    # # OS-specific script extension (modified later if we're running on Windows)
    script_extension = '.sh'

    # If False, we'll load chunk files with file lists if they exist
    force_enumeration = False

    # Prefer threads on Windows, processes on Linux
    parallelization_defaults_to_threads = False

    # This is for things like image rendering, not for MegaDetector
    default_workers_for_parallel_tasks = 30

    overwrite_handling = 'skip' # 'skip', 'error', or 'overwrite'

    # The function used to get camera names from image paths; can also replace this
    # with a custom function.
    relative_path_to_location = image_file_to_camera_folder

    # This will be the .json results file after RDE; if this is still None when
    # we get to classification stuff, that will indicate that we didn't do RDE.
    filtered_output_filename = None

    if os.name == 'nt':

        slcc = '^'
        scc = 'REM'
        script_extension = '.bat'

        # My experience has been that Python multiprocessing is flaky on Windows, so
        # default to threads on Windows
        parallelization_defaults_to_threads = True
        default_workers_for_parallel_tasks = 10

    ## Constants related to using YOLOv5's val.py

    # Should we use YOLOv5's val.py instead of run_detector_batch.py?
    use_yolo_inference_scripts = False

    # Directory in which to run val.py (relevant for YOLOv5, not for YOLOv8)
    yolo_working_dir = os.path.expanduser('~/git/yolov5')

    # Only used for loading the mapping from class indices to names
    yolo_dataset_file = None

    # 'yolov5' or 'yolov8'; assumes YOLOv5 if this is None
    yolo_model_type = None

    # inference batch size
    yolo_batch_size = 1

    # Should we remove intermediate files used for running YOLOv5's val.py?
    # Only relevant if use_yolo_inference_scripts is True.
    remove_yolo_intermediate_results = True
    remove_yolo_symlink_folder = True
    use_symlinks_for_yolo_inference = True
    write_yolo_debug_output = False

    # Should we apply YOLOv5's test-time augmentation?
    augment = False

    ## Constants related to tiled inference
    use_tiled_inference = False

    # Should we delete tiles after each job?  Only set this to False for debugging;
    # large jobs will take up a lot of space if you keep tiles around after each task.
    remove_tiles = True
    tile_size = (1280,1280)
    tile_overlap = 0.2

    #################################################################################################
    # JOB-SPECIFIC CONSTANTS                                                                        #
    #################################################################################################

    assert not (input_path.endswith('/') or input_path.endswith('\\'))
    assert os.path.isdir(input_path), 'Could not find input folder {}'.format(input_path)
    input_path = input_path.replace('\\','/')
    assert job_date is not None and job_name != 'organization'

    # Optional descriptor
    job_tag = None

    if job_tag is None:
        job_description_string = ''
    else:
        job_description_string = '-' + job_tag

    model_file = 'MDV5A' # 'MDV5A', 'MDV5B', 'MDV4'

    postprocessing_base = f"{os.getcwd()}/postprocessing"
    if verbose: print(f"postprocessing_base: {postprocessing_base}", flush=True)

    # Set to "None" when using augmentation or an image queue, which don't currently support
    # checkpointing.  Don't worry, this will be assert()'d in the next cell.
    checkpoint_frequency = 10000

    # Estimate inference speed for the current GPU
    approx_images_per_second = estimate_md_images_per_second(model_file)

    # Rough estimate for the inference time cost of augmentation
    if augment and (approx_images_per_second is not None):
        approx_images_per_second = approx_images_per_second * 0.7

    base_task_name = job_name + '-' + job_date + job_description_string + '-' + \
        get_detector_version_from_filename(model_file)
    base_output_folder_name = os.path.join(postprocessing_base,job_name)
    os.makedirs(base_output_folder_name,exist_ok=True)

    #################################################################################################
    # DERIVED VARIABLES, CONSTANT VALIDATION, PATH SETUP                                            #
    #################################################################################################

    if use_image_queue:
        assert checkpoint_frequency is None,\
            'Checkpointing is not supported when using an image queue'

    if augment:
        assert checkpoint_frequency is None,\
            'Checkpointing is not supported when using augmentation'

        assert use_yolo_inference_scripts,\
            'Augmentation is only supported when running with the YOLO inference scripts'

    if use_tiled_inference:
        assert not augment, \
            'Augmentation is not supported when using tiled inference'
        assert not use_yolo_inference_scripts, \
            'Using the YOLO inference script is not supported when using tiled inference'
        assert checkpoint_frequency is None, \
            'Checkpointing is not supported when using tiled inference'

    filename_base = os.path.join(base_output_folder_name, base_task_name)
    combined_api_output_folder = os.path.join(filename_base, 'combined_api_outputs')
    postprocessing_output_folder = os.path.join(filename_base, 'preview')

    combined_api_output_file = os.path.join(
        combined_api_output_folder,
        '{}_detections.json'.format(base_task_name))

    os.makedirs(filename_base, exist_ok=True)
    os.makedirs(combined_api_output_folder, exist_ok=True)
    os.makedirs(postprocessing_output_folder, exist_ok=True)

    if input_path.endswith('/'):
        input_path = input_path[0:-1]

    if verbose: print(f'Output folder:\n{filename_base}', flush=True)

    #################################################################################################
    # ENUMERATE FILES FOR FUTURE USE                                                                #
    #################################################################################################

    # Have we already listed files for this job?
    chunk_files = os.listdir(filename_base)
    pattern = re.compile('chunk\d+.json')
    chunk_files = [fn for fn in chunk_files if pattern.match(fn)] # generated in cells below, if this does not exist

    if (not force_enumeration) and (len(chunk_files) > 0):

        if verbose: print('Found {} chunk files in folder {}, bypassing enumeration'.format(
            len(chunk_files),
            filename_base))

        all_images = []
        for fn in chunk_files:
            with open(os.path.join(filename_base,fn),'r') as f:
                chunk = json.load(f)
                assert isinstance(chunk,list)
                all_images.extend(chunk)
        all_images = sorted(all_images)

        if verbose: print('Loaded {} image files from {} chunks in {}'.format(
            len(all_images),len(chunk_files),filename_base))

    else:

        if verbose: print('Enumerating image files in {}'.format(input_path))

        all_images = sorted(path_utils.find_images(input_path,recursive=True,convert_slashes=True))

        # It's common to run this notebook on an external drive with the main folders in the drive root
        all_images = [fn for fn in all_images if not \
                    (fn.startswith('$RECYCLE') or fn.startswith('System Volume Information'))]

        if verbose: print('')

        if verbose: print('Enumerated {} image files in {}'.format(len(all_images),input_path))

    if verbose: print(flush=True)

    #################################################################################################
    # DIVIDE IMAGES INTO CHUNKS FOR MULTIPLE PROCESSES                                              #
    #################################################################################################

    folder_chunks = split_list_into_n_chunks(all_images,n_jobs)

    #################################################################################################
    # ESTIMATE TOTAL TIME                                                                           #
    #################################################################################################

    if approx_images_per_second is None:

        if verbose: print("Can't estimate inference time for the current environment")

    else:

        n_images = len(all_images)
        execution_seconds = n_images / approx_images_per_second
        wallclock_seconds = execution_seconds / n_gpus
        if verbose: print('Expected time: {}'.format(humanfriendly.format_timespan(wallclock_seconds)))

        seconds_per_chunk = len(folder_chunks[0]) / approx_images_per_second
        if verbose: print('Expected time per chunk: {}'.format(humanfriendly.format_timespan(seconds_per_chunk)))

    if verbose: print(flush=True)

    #################################################################################################
    # WRITE FILE LISTS                                                                              #
    #################################################################################################

    task_info = []

    for i_chunk, chunk_list in enumerate(folder_chunks):

        chunk_fn = os.path.join(filename_base,'chunk{}.json'.format(str(i_chunk).zfill(3)))
        task_info.append({'id':i_chunk,'input_file':chunk_fn})
        path_utils.write_list_to_file(chunk_fn, chunk_list)

    #################################################################################################
    # GENERATE COMMANDS                                                                             #
    #################################################################################################

    # A list of the scripts tied to each GPU, as absolute paths.  We'll write this out at
    # the end so each GPU's list of commands can be run at once
    gpu_to_scripts = defaultdict(list)

    # i_task = 0; task = task_info[i_task]
    for i_task,task in enumerate(task_info):

        chunk_file = task['input_file']
        checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')

        output_fn = chunk_file.replace('.json','_results.json')

        task['output_file'] = output_fn

        if n_gpus > 1:
            gpu_number = i_task % n_gpus
        else:
            gpu_number = default_gpu_number

        image_size_string = ''
        if image_size is not None:
            image_size_string = '--image_size {}'.format(image_size)

        # Generate the script to run MD

        if use_yolo_inference_scripts:

            augment_string = ''
            if augment:
                augment_string = '--augment_enabled 1'
            else:
                augment_string = '--augment_enabled 0'

            batch_string = '--batch_size {}'.format(yolo_batch_size)

            symlink_folder = os.path.join(filename_base,'symlinks','symlinks_{}'.format(
                str(i_task).zfill(3)))
            yolo_results_folder = os.path.join(filename_base,'yolo_results','yolo_results_{}'.format(
                str(i_task).zfill(3)))

            symlink_folder_string = '--symlink_folder "{}"'.format(symlink_folder)
            yolo_results_folder_string = '--yolo_results_folder "{}"'.format(yolo_results_folder)

            remove_symlink_folder_string = ''
            if not remove_yolo_symlink_folder:
                remove_symlink_folder_string = '--no_remove_symlink_folder'

            write_yolo_debug_output_string = ''
            if write_yolo_debug_output:
                write_yolo_debug_output = '--write_yolo_debug_output'

            remove_yolo_results_string = ''
            if not remove_yolo_intermediate_results:
                remove_yolo_results_string = '--no_remove_yolo_results_folder'

            confidence_threshold_string = ''
            if json_threshold is not None:
                confidence_threshold_string = '--conf_thres {}'.format(json_threshold)
            else:
                confidence_threshold_string = '--conf_thres {}'.format(DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD)

            cmd = ''

            device_string = '--device {}'.format(gpu_number)

            overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)

            cmd += f'python run_inference_with_yolov5_val.py "{model_file}" "{chunk_file}" "{output_fn}" '
            cmd += f'{image_size_string} {augment_string} '
            cmd += f'{symlink_folder_string} {yolo_results_folder_string} {remove_yolo_results_string} '
            cmd += f'{remove_symlink_folder_string} {confidence_threshold_string} {device_string} '
            cmd += f'{overwrite_handling_string} {batch_string} {write_yolo_debug_output_string}'

            if yolo_working_dir is not None:
                cmd += f' --yolo_working_folder "{yolo_working_dir}"'
            if yolo_dataset_file is not None:
                cmd += ' --yolo_dataset_file "{}"'.format(yolo_dataset_file)
            if yolo_model_type is not None:
                cmd += ' --model_type {}'.format(yolo_model_type)

            if not use_symlinks_for_yolo_inference:
                cmd += ' --no_use_symlinks'

            cmd += '\n'

        elif use_tiled_inference:

            tiling_folder = os.path.join(filename_base,'tile_cache','tile_cache_{}'.format(
                str(i_task).zfill(3)))

            if os.name == 'nt':
                cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
            else:
                cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '

            cmd = f'{cuda_string} python run_tiled_inference.py "{model_file}" "{input_path}" "{tiling_folder}" "{output_fn}"'

            cmd += f' --image_list "{chunk_file}"'
            cmd += f' --overwrite_handling {overwrite_handling}'

            if not remove_tiles:
                cmd += ' --no_remove_tiles'

            # If we're using non-default tile sizes
            if tile_size is not None and (tile_size[0] > 0 or tile_size[1] > 0):
                cmd += ' --tile_size_x {} --tile_size_y {}'.format(tile_size[0],tile_size[1])

            if tile_overlap is not None:
                cmd += f' --tile_overlap {tile_overlap}'

        else:

            if os.name == 'nt':
                cuda_string = f'set CUDA_VISIBLE_DEVICES={gpu_number} & '
            else:
                cuda_string = f'CUDA_VISIBLE_DEVICES={gpu_number} '

            checkpoint_frequency_string = ''
            checkpoint_path_string = ''

            if checkpoint_frequency is not None and checkpoint_frequency > 0:
                checkpoint_frequency_string = f'--checkpoint_frequency {checkpoint_frequency}'
                checkpoint_path_string = '--checkpoint_path "{}"'.format(checkpoint_filename)

            use_image_queue_string = ''
            if (use_image_queue):
                use_image_queue_string = '--use_image_queue'

            ncores_string = ''
            if (ncores > 1):
                ncores_string = '--ncores {}'.format(ncores)

            quiet_string = ''
            if quiet_mode:
                quiet_string = '--quiet'

            confidence_threshold_string = ''
            if json_threshold is not None:
                confidence_threshold_string = '--threshold {}'.format(json_threshold)

            overwrite_handling_string = '--overwrite_handling {}'.format(overwrite_handling)
            cmd = f'{cuda_string} python run_detector_batch.py "{model_file}" "{chunk_file}" "{output_fn}" {checkpoint_frequency_string} {checkpoint_path_string} {use_image_queue_string} {ncores_string} {quiet_string} {image_size_string} {confidence_threshold_string} {overwrite_handling_string}'

            if include_image_size:
                cmd += ' --include_image_size'
            if include_image_timestamp:
                cmd += ' --include_image_timestamp'
            if include_exif_data:
                cmd += ' --include_exif_data'

        cmd_file = os.path.join(filename_base,'run_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                                str(gpu_number).zfill(2),script_extension))

        with open(cmd_file,'w') as f:
            f.write(cmd + '\n')

        st = os.stat(cmd_file)
        os.chmod(cmd_file, st.st_mode | stat.S_IEXEC)

        task['command'] = cmd
        task['command_file'] = cmd_file

        # Generate the script to resume from the checkpoint (only supported with MD inference code)

        gpu_to_scripts[gpu_number].append(cmd_file)

        if checkpoint_frequency is not None:

            resume_string = ' --resume_from_checkpoint "{}"'.format(checkpoint_filename)
            resume_cmd = cmd + resume_string

            resume_cmd_file = os.path.join(filename_base,
                                        'resume_chunk_{}_gpu_{}{}'.format(str(i_task).zfill(3),
                                        str(gpu_number).zfill(2),script_extension))

            with open(resume_cmd_file,'w') as f:
                f.write(resume_cmd + '\n')

            st = os.stat(resume_cmd_file)
            os.chmod(resume_cmd_file, st.st_mode | stat.S_IEXEC)

            task['resume_command'] = resume_cmd
            task['resume_command_file'] = resume_cmd_file

    # ...for each task

    # Write out a script for each GPU that runs all of the commands associated with
    # that GPU.  Typically only used when running lots of little scripts in lieu
    # of checkpointing.
    for gpu_number in gpu_to_scripts:

        gpu_script_file = os.path.join(filename_base,'run_all_for_gpu_{}{}'.format(
            str(gpu_number).zfill(2),script_extension))
        with open(gpu_script_file,'w') as f:
            for script_name in gpu_to_scripts[gpu_number]:
                s = script_name
                # When calling a series of batch files on Windows from within a batch file, you need to
                # use "call", or only the first will be executed.  No, it doesn't make sense.
                if os.name == 'nt':
                    s = 'call ' + s
                f.write(s + '\n')
            f.write('echo "Finished all commands for GPU {}"'.format(gpu_number))
        st = os.stat(gpu_script_file)
        os.chmod(gpu_script_file, st.st_mode | stat.S_IEXEC)

    # ...for each GPU

    #################################################################################################
    # RUN THE TASKS                                                                                 #
    #################################################################################################

    # Dan Morris's notes: 

    # The cells we've run so far wrote out some shell scripts (.bat files on Windows,
    # .sh files on Linx/Mac) that will run MegaDetector.  I like to leave the interactive
    # environment at this point and run those scripts at the command line.  So, for example,
    # if you're on Windows, and you've basically used the default values above, there will be
    # batch files called, e.g.:

    # c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_000_gpu_00.bat
    # c:\users\[username]\postprocessing\[organization]\[job_name]\run_chunk_001_gpu_01.bat

    # Those batch files expect to be run from the "detection" folder of the MegaDetector repo,
    # typically:

    # c:\git\MegaDetector\megadetector\detection

    # All of that said, you don't *have* to do this at the command line.  The following code
    # runs these scripts programmatically, so if you set "run_md_automatically" to "True",
    # you can run MegaDetector without leaving this notebook.

    # One downside of the programmatic approach is that this code doesn't yet parallelize over
    # multiple processes, so the tasks will run serially.  This only matters if you have
    # multiple GPUs.

    if run_md_automatically:

        assert not use_yolo_inference_scripts, \
            'If you want to use the YOLOv5 inference scripts, you can\'t run the model interactively (yet)'

        # i_task = 0; task = task_info[i_task]
        for i_task,task in enumerate(task_info):

            chunk_file = task['input_file']
            output_fn = task['output_file']

            checkpoint_filename = chunk_file.replace('.json','_checkpoint.json')

            if json_threshold is not None:
                confidence_threshold = json_threshold
            else:
                confidence_threshold = DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD

            if checkpoint_frequency is not None and checkpoint_frequency > 0:
                cp_freq_arg = checkpoint_frequency
            else:
                cp_freq_arg = -1

            start_time = time.time()
            results = load_and_run_detector_batch(model_file=model_file,
                                                image_file_names=chunk_file,
                                                checkpoint_path=checkpoint_filename,
                                                confidence_threshold=confidence_threshold,
                                                checkpoint_frequency=cp_freq_arg,
                                                results=None,
                                                n_cores=ncores,
                                                use_image_queue=use_image_queue,
                                                quiet=quiet_mode,
                                                image_size=image_size)
            elapsed = time.time() - start_time

            if verbose: print('Task {}: finished inference for {} images in {}'.format(
                i_task, len(results),humanfriendly.format_timespan(elapsed)))

            # This will write absolute paths to the file, we'll fix this later
            write_results_to_file(results, output_fn, detector_file=model_file)

            if checkpoint_frequency is not None and checkpoint_frequency > 0:
                if os.path.isfile(checkpoint_filename):
                    os.remove(checkpoint_filename)
                    if verbose: print('Deleted checkpoint file {}'.format(checkpoint_filename))

        # ...for each chunk

    # ...if we're running tasks in this notebook

    if verbose: print(flush=True)

    #################################################################################################
    # LOAD RESULTS                                                                                  #
    #################################################################################################

    # look for failed or missing images in each task by checking that all task output files exist
    missing_output_files = []

    # i_task = 0; task = task_info[i_task]
    for i_task, task in tqdm(enumerate(task_info),total=len(task_info)):
        output_file = task['output_file']
        if not os.path.isfile(output_file):
            missing_output_files.append(output_file)

    if len(missing_output_files) > 0:
        if verbose: print('Missing {} output files:'.format(len(missing_output_files)))
        for s in missing_output_files:
            if verbose: print(s)
        raise Exception('Missing output files')


    n_total_failures = 0

    for i_task,task in tqdm(enumerate(task_info),total=len(task_info)):

        chunk_file = task['input_file']
        output_file = task['output_file']

        with open(chunk_file,'r') as f:
            task_images = json.load(f)
        with open(output_file,'r') as f:
            task_results = json.load(f)

        task_images_set = set(task_images)
        filename_to_results = {}

        n_task_failures = 0

        for im in task_results['images']:
            # Most of the time, inference result files use absolute paths, but it's
            # getting annoying to make sure that's *always* true, so handle both here.
            # E.g., when using tiled inference, paths will be relative.
            if not os.path.isabs(im['file']):
                fn = os.path.join(input_path,im['file']).replace('\\','/')
                im['file'] = fn
            assert im['file'].startswith(input_path)
            assert im['file'] in task_images_set
            filename_to_results[im['file']] = im
            if 'failure' in im:
                assert im['failure'] is not None
                n_task_failures += 1

        task['n_failures'] = n_task_failures
        task['results'] = task_results

        for fn in task_images:
            assert fn in filename_to_results, \
                'File {} not found in results for task {}'.format(fn,i_task)

        n_total_failures += n_task_failures

    # ...for each task

    assert n_total_failures < max_tolerable_failed_images,\
        '{} failures (max tolerable set to {})'.format(n_total_failures,
                                                    max_tolerable_failed_images)

    if verbose: print('Processed all {} images with {} failures'.format(
        len(all_images),n_total_failures))


    ##%% Merge results files and make filenames relative

    combined_results = {}
    combined_results['images'] = []
    images_processed = set()

    for i_task,task in tqdm(enumerate(task_info),total=len(task_info)):

        task_results = task['results']

        if i_task == 0:
            combined_results['info'] = task_results['info']
            combined_results['detection_categories'] = task_results['detection_categories']
        else:
            assert task_results['info']['format_version'] == combined_results['info']['format_version']
            assert task_results['detection_categories'] == combined_results['detection_categories']

        # Make sure we didn't see this image in another chunk
        for im in task_results['images']:
            assert im['file'] not in images_processed
            images_processed.add(im['file'])

        combined_results['images'].extend(task_results['images'])

    # Check that we ended up with the right number of images
    assert len(combined_results['images']) == len(all_images), \
        'Expected {} images in combined results, found {}'.format(
            len(all_images),len(combined_results['images']))

    # Check uniqueness
    result_filenames = [im['file'] for im in combined_results['images']]
    assert len(combined_results['images']) == len(set(result_filenames))

    # Convert to relative paths, preserving '/' as the path separator, regardless of OS
    for im in combined_results['images']:
        assert '\\' not in im['file']
        assert im['file'].startswith(input_path)
        if input_path.endswith(':'):
            im['file'] = im['file'].replace(input_path,'',1)
        else:
            im['file'] = im['file'].replace(input_path + '/','',1)

    with open(combined_api_output_file,'w') as f:
        json.dump(combined_results,f,indent=1)

    if verbose: print('Wrote results to {}'.format(combined_api_output_file))

    if verbose: print(flush=True)

    #################################################################################################
    # POST-PROCESSING (PRE-RDE)                                                                     #
    #################################################################################################

    render_animals_only = False

    options = PostProcessingOptions()
    options.image_base_dir = input_path
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.2
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    options.sample_seed = 0
    options.max_figures_per_html_file = 2500

    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
    options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

    if render_animals_only:
        # Omit some pages from the output, useful when animals are rare
        options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                        'detections_person_vehicle','non_detections']

    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{:.3f}'.format(options.confidence_threshold))
    if render_animals_only:
        output_base = output_base + '_animals_only'

    os.makedirs(output_base, exist_ok=True)
    if verbose: print('Processing to {}'.format(output_base))

    options.md_results_file = combined_api_output_file
    options.output_dir = output_base

    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')

    #################################################################################################
    # FIND REPEAT DETECTION                                                                         #
    #################################################################################################

    task_index = 0

    options = repeat_detections_core.RepeatDetectionOptions()

    options.confidenceMin = 0.1
    options.confidenceMax = 1.01
    options.iouThreshold = 0.85
    options.occurrenceThreshold = 15
    options.maxSuspiciousDetectionSize = 0.2
    # options.minSuspiciousDetectionSize = 0.05

    options.parallelizationUsesThreads = parallelization_defaults_to_threads
    options.nWorkers = default_workers_for_parallel_tasks

    # This will cause a very light gray box to get drawn around all the detections
    # we're *not* considering as suspicious.
    options.bRenderOtherDetections = True
    options.otherDetectionsThreshold = options.confidenceMin

    options.bRenderDetectionTiles = True
    options.maxOutputImageWidth = 2000
    options.detectionTilesMaxCrops = 250

    # options.lineThickness = 5
    # options.boxExpansion = 8

    # To invoke custom collapsing of folders for a particular manufacturer's naming scheme
    options.customDirNameFunction = relative_path_to_location

    options.bRenderHtml = False
    options.imageBase = input_path
    rde_string = 'rde_{:.3f}_{:.3f}_{}_{:.3f}'.format(
        options.confidenceMin, options.iouThreshold,
        options.occurrenceThreshold, options.maxSuspiciousDetectionSize)
    options.outputBase = os.path.join(filename_base, rde_string + '_task_{}'.format(task_index))
    options.filenameReplacements = None # {'':''}

    # Exclude people and vehicles from RDE
    # options.excludeClasses = [2,3]

    # options.maxImagesPerFolder = 50000
    # options.includeFolders = ['a/b/c']
    # options.excludeFolder = ['a/b/c']

    options.debugMaxDir = -1
    options.debugMaxRenderDir = -1
    options.debugMaxRenderDetection = -1
    options.debugMaxRenderInstance = -1

    # Can be None, 'xsort', or 'clustersort'
    options.smartSort = 'xsort'

    # run detection 
    suspicious_detection_results = repeat_detections_core.find_repeat_detections(combined_api_output_file, outputFilename=None, options=options)

    #################################################################################################
    # MANUAL RDE STEP                                                                               #
    ################################################################################################### DELETE THE VALID DETECTIONS ##

    # If you run this line, it will open the folder up in your file browser
    path_utils.open_file(os.path.dirname(suspicious_detection_results.filterFile),attempt_to_open_in_wsl_host=True)

    return combined_api_output_file, rde_string, suspicious_detection_results, default_workers_for_parallel_tasks, parallelization_defaults_to_threads, postprocessing_output_folder, base_task_name

def post_rde(input_path, 
            combined_api_output_file, rde_string, suspicious_detection_results, 
            default_workers_for_parallel_tasks, parallelization_defaults_to_threads, 
            postprocessing_output_folder, base_task_name, verbose=False):
            
    '''
    Run post-RDE processing. 

    NOTE: 
    Except for input_path and verbose, the user does not have to understand what the arguments 
    are - they are the outputs returned by mdv5a_and_rde(), also defined in this script. 

    Args:
        input_path (str): Path to data folder, e.g., '/home/user/data'
        combined_api_output_file (str): filepath to megedetection results on the whole dataset 
        rde_string (str): string containing rde settings 
        suspicious_detection_results (object): repeat detection results 
        default_workers_for_parallel_tasks (int): settings for image rendering
        parallelization_defaults_to_threads (bool): prefer threads on Windows, processes on Linux
        postprocessing_output_folder (str): folder path to filtered output 
        base_task_name (str): naming conventions 
        verbose (bool): Whether to print job details 

    Return: 
        filtered_output_filename (str): path to the output file
    '''

    filtered_output_filename = path_utils.insert_before_extension(combined_api_output_file,
                                                              'filtered_{}'.format(rde_string))
    remove_repeat_detections.remove_repeat_detections(
        inputFile=combined_api_output_file,
        outputFile=filtered_output_filename,
        filteringDir=os.path.dirname(suspicious_detection_results.filterFile)
        )
    
    render_animals_only = False

    options = PostProcessingOptions()
    options.image_base_dir = input_path
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.2
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    options.sample_seed = 0
    options.max_figures_per_html_file = 5000

    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = default_workers_for_parallel_tasks
    options.parallelize_rendering_with_threads = parallelization_defaults_to_threads

    if render_animals_only:
        # Omit some pages from the output, useful when animals are rare
        options.rendering_bypass_sets = ['detections_person','detections_vehicle',
                                        'detections_person_vehicle','non_detections']

    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{}_{:.3f}'.format(rde_string, options.confidence_threshold))

    if render_animals_only:
        output_base = output_base + '_render_animals_only'
    os.makedirs(output_base, exist_ok=True)

    if verbose: print('Processing post-RDE to {}'.format(output_base))

    options.md_results_file = filtered_output_filename
    options.output_dir = output_base

    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')

    if verbose: print('Done')

    return filtered_output_filename

def upload_db(df, dataset_name, table_name):
    '''
    Upload the given dataset to the database under dataset_name.

    Args:
        df (pandas dataframe): dataset data (as produced by the preprocessing notebooks)
        dataset_name (str): value for the 'dataset' column in the database 
    '''
    # Database credentials and settings
    db_user = 'dataprep'
    db_pass = 'sdK77:+,^^g[+rbV'
    db_name = 'images'
    db_ip   = '127.0.0.1:2235'  # Corrected IP address

    # Connection URL
    URL = f'mysql+pymysql://{db_user}:{db_pass}@{db_ip}/{db_name}'

    # Create the SQLAlchemy engine
    engine = sqlalchemy.create_engine(URL, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)

    # Update the given dataset name
    df['dataset'] = dataset_name

    # Upload to database 
    chunksize = 100000
    for i in tqdm(range(0, len(df), chunksize)):
        while 1:
            try:
                df.iloc[i:i+chunksize].to_sql(table_name, con=engine, if_exists='append', index=False)
                break
            except Exception as e:
                print(e)

def read_db(dataset_name, table_name):
    random.seed(42)

    # Database credentials and settings
    db_user = 'dataprep'
    db_pass = 'sdK77:+,^^g[+rbV'
    db_name = 'images'
    db_ip   = '127.0.0.1:2235'  # Corrected IP address

    # Connection URL
    URL = f'mysql+pymysql://{db_user}:{db_pass}@{db_ip}/{db_name}'

    # Create the SQLAlchemy engine
    engine = sqlalchemy.create_engine(URL, pool_size=5, max_overflow=2, pool_timeout=30, pool_recycle=1800)
    print('Engine Created')

    # Retrieve and return the dataset
    openesc, closeesc = '', '' # escape column names
    query = f'SELECT * FROM images.{table_name}'
    query += f' WHERE {openesc}dataset{closeesc} = "{dataset_name}"'

    return pd.read_sql(query, con=engine)