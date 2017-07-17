'''
A robust set of tools for visualizing flow structures using the maximum z projection technique

Developed by William Gilpin, Vivek Prakash, and Manu Prakash, 2015

http://williamgilpin.github.io/flowtrace_docs/flowtrace_python.html

Future
------

+ Once skimage.io is improved, switch to using this for most of the file reading
and writing capabilities
+ File writing needs to be sped up a lot
+ Try to modularize some functions (like the array roller) with fixed call signatures, then compile them with numba 
+ Find a concise kwarg structure so that  the user to specify colors for time-coding
+ Allow direct invocation from command line
+ Add more robust protection to prevent overloading CPU when multithreading. Get the total RAM available
    and compare it to the total RAM required per CPU used for stacks of a given length
'''
from numpy import *
import warnings
import os
import glob


from scipy.misc import toimage
from scipy.misc import imsave

try:
    from scipy.ndimage.morphology import grey_dilation
except ImportError:
    warn('scipy.ndimage.morphology not imported, this may be required in future versions')

# try:
#     from numba import jit
#     numba_available=True
# except ImportError:
#     numba_available=False
#     warn('numba not imported, code might run slightly slower on large datasets')

try:
    from scipy.misc import imread as imread2 
except ImportError:
    warn('scipy.misc not imported, using slower numpy.imread')
    from numpy import imread as imread2 

try:
    import multiprocessing as mp
    parallel_available = True
except ImportError:
    parallel_available = False
    warnings.warn('No multiprocessing module found, running code on single processor only')

def flowtrace(image_dir, frames_to_merge, out_dir='same', use_parallel=True, max_cores=8, frames_to_skip=1, **kwargs):
    '''
    This code is optimized for computers that DO NOT have enough 
    RAM to store the entire time series in memory, but which DO have enough 
    RAM to store an entire z-projection stack in memory. When using multithreading,
    be sure to multiply the memory usage per stack by the number of cores that will be used.
    
    Parameters
    ----------
    image_dir : str
        path to directory containing raw image files
        
    frames_to_merge : int
        The numbe of frames to merge to form a single streamline image
    
    out_dir : str
        path to directory where streamline files are saved
        
    max_cores : int
        With multi-threading enabled, the maximum number of simultaneous jobs
        to run. When running this, it's useful to calculate the total RAM available
        versus that required to completely hold an image stack of length frames_to_merge,
        since parallelization is not useful if there's not enough RAM for each core to
        complete a task

    frames_to_skip : int
        The number of images to skip when building each substack

    kwargs
    ------

    take_diff : bool
        whether to take the difference of consecutive frames
    
    diff_order : int
        the order of the difference in frames when take_diff is used

    subtract_median : bool
        For each substack, subtract the median before taking the 
        z projection

    subtract_first : bool
        For each substack, subtract the first frame before taking the 
        z projection

    add_first_frame : bool
        Add the unaltered first frame of the stack back to the stack
        before taking z projection. Makes it possible to view sharp structures
        in median transformed data

    color_series : bool
        Color the time traces
        
    William Gilpin, Vivek Prakash, and Manu Prakash, 2015
    '''
    image_files = glob.glob(image_dir+'/*.tif')
    image_files.sort()  # sort the glob list
    image_files = image_files[::frames_to_skip]
    im_path = os.path.split(image_files[0])[:-1][0]
    if out_dir == 'same':
        out_dir = image_dir
    
    
    if use_parallel and parallel_available:

        # Set up a list of processes that we want to run
        num_cores = mp.cpu_count()
        
        num_cores = min([num_cores, max_cores])

        nframes = len(image_files)

        runlen = ceil(nframes/num_cores) # think about whether this will throw an error and floor is safer
        while runlen < frames_to_merge:
            warnings.warn('Number of active cores was reduced by one')
            num_cores -= 1
            runlen = ceil(nframes/num_cores)

        runlen = int(runlen)
        chunks_list = [image_files[ii:ii + runlen] for ii in range(0, len(image_files), runlen)]

        if len(chunks_list[-1])<frames_to_merge:
            last_chunk = chunks_list.pop()
            chunks_list[-1]+=last_chunk

        for ind, chunk in enumerate(chunks_list):
            if ind==(len(chunks_list)-1):
                continue
            chunk+=(chunks_list[ind+1][:frames_to_merge])

        # now run parallel processes
        processes = [mp.Process(target=sliding_zproj_internal, args=(chunks_list[ii], frames_to_merge, out_dir), kwargs=kwargs) for ii in range(num_cores)]
        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
    
    else:
        sliding_zproj_internal(image_files, frames_to_merge, out_dir, **kwargs)


def sliding_zproj_internal(frames_list, frames_to_merge, out_dir, **kwargs):
    '''
    This code is optimized for computers that DO NOT have enough 
    RAM to store all the images in memory, but which DO have enough 
    RAM to store an entire z-projection stack in memory
    
    Parameters
    ----------
    frames_list : list of str
        list of image files
        
    frames_to_merge : int
        The number of frames to merge to form a single streamline image
    
    out_dir : str
        path to directory where streamline files are saved
        
    take_diff : bool
        whether to take the difference of consecutive frames
    
    diff_order : int
        the order of the difference in frames when take_diff is used

    subtract_median : bool
        For each substack, subtract the median before taking the 
        z projection

    subtract_first : bool
        For each substack, subtract the first frame before taking the 
        z projection

    add_first_frame : bool
        Add the unaltered first frame of the stack back to the stack
        before taking z projection. Makes it possible to view sharp structures
        in median transformed data

    invert_color : bool
        Set to True if working with dark particles on a light background

    color_series : bool
        Color the time traces


        
    William Gilpin, Vivek Prakash, and Manu Prakash, 2015
    '''

    if 'diff_order' in kwargs:
        diff_order = kwargs['diff_order']
    else:
        diff_order = 1


    image_files = frames_list
    im_path = os.path.split(image_files[0])[:-1][0]

    frame0 = imread2(image_files[0])

    if len(frame0.shape)==3:
        rgb_flag=True
        frame0 = sum(frame0, axis=2)/3.
    else:
        rgb_flag=False

    # preallocate storage array
    if rgb_flag:
        stack_shape = frame0.shape+(3,)+(frames_to_merge,)
    else:
        stack_shape = frame0.shape+(frames_to_merge,)
    stack = zeros(stack_shape)


    for ii in range(len(image_files)-frames_to_merge):

        if ii == 0:    
            for jj in range(frames_to_merge):
                im = imread2(image_files[jj])
                stack[...,jj] = im
        else:
            if rgb_flag:
                stack = roll(stack, -1, axis=3)
            else:
                stack = roll(stack, -1, axis=2)
            im = imread2(image_files[ii+frames_to_merge])
            stack[...,-1] = im

        stack2 = stack.copy()


        if 'subtract_first' in kwargs:
            if kwargs['subtract_first']:
                front_im = stack2[...,0]
                stack2 = stack2-front_im[..., newaxis]
        if 'subtract_median' in kwargs:
            if kwargs['subtract_median']:
                med_im= median(stack2, axis=-1)
                stack2 = stack2-med_im[..., newaxis]
        if 'take_diff' in kwargs:
            if kwargs['take_diff']:
                stack2 = diff(stack2, n=diff_order, axis=-1)

        if 'add_first_frame' in kwargs:        
            if kwargs['add_first_frame']:
                stack2 = dstack([stack2, stack[...,0]])
        
        if 'color_series' in kwargs:
            if kwargs['color_series']:

                # probably want to preallocate this upstream (255, 204, 153)
                # fullcmap = array(cmap1D((0,250,154),(255,20,147), stack2.shape[-1]))/255.
                fullcmap = array(cmap1D((90, 10, 250),(255, 153, 0),stack2.shape[-1]))/255.

                rvals, gvals, bvals = stack2*fullcmap[:, 0], stack2*fullcmap[:, 1], stack2*fullcmap[:, 2]
                stack2 = concatenate([rvals[...,newaxis],gvals[...,newaxis],bvals[...,newaxis]],axis=-1)
                stack2 = swapaxes(stack2,-1,-2)


        if 'invert_color' in kwargs:
            if 'color_series' in kwargs:
                warnings.warn("Be careful when enabling time series coloring with images with light backgrounds")
            if kwargs['invert_color']:
                max_proj = stack2.min(axis=-1)
                max_proj = 255 - max_proj

            else:
                max_proj = stack2.max(axis=-1)
        else:
            max_proj = stack2.max(axis=-1)


        im_name = os.path.split(image_files[ii])[-1][:-4]

        savestr = out_dir +'/'+im_name+'_streamlines'+'_frames'+str(frames_to_merge)+'.png'

        # This makes it difficult to save with a light background, and it also might cause flickering
        # due to re-normalization
        toimage(max_proj, cmin=0.0, cmax=max(ravel(max_proj))).save(savestr)


def overlay_images(dir1, dir2, out_dir, ftype1='.tif',ftype2='.tif'):
    '''
    Given two directories full of images, load one image from each directory and 
    overlay it on the other with the specified tint and color

    Parameters
    ----------

    dir1 : str
        Path to the directory of images that will be used as the background

    dir2 : str
        Path to the directory of images that will be tinted and overlaid

    out_dir : str
        Path to the directory at which output will be saved

    '''
    tint_color = (127/255., 255/255., 212/255.)

# rvals, gvals, bvals = stack2*fullcmap[:, 0], stack2*fullcmap[:, 1], stack2*fullcmap[:, 2]
#                 stack2 = concatenate([rvals[...,newaxis],gvals[...,newaxis],bvals[...,newaxis]],axis=-1)

    bg_ims = glob.glob(dir1+'/*'+ftype1)
    fg_ims = glob.glob(dir2+'/*'+ftype2)

    if len(bg_ims) != len(bg_ims):
        warnings.warn("The two image directories contain different numbers of images.")

    for ind, bg_im in enumerate(bg_ims):

        fg_im = fg_ims[ind]
        im1 = imread2(bg_im)
        im2 = imread2(fg_im)

        # threshold image from one side
        # ones just pass through, anything less than one filters
        im2[im2 < .5] = 1.0

        finim = im1*im2

        bg_name = os.path.split(bg_im)[-1][:-4]
        fg_name = os.path.split(fg_im)[-1][:-4]
        savestr = out_dir +'/'+bg_name+'_times_'+fg_name+'.png'
        toimage(finim, cmin=0.0, cmax=max(ravel(finim))).save(savestr)


def cmap1D(col1, col2, N):
    '''Generate a continuous colormap between two values
    
    Parameters
    ----------
    
    col1 : tuple of ints
        RGB values of final color
        
    col2 : tuple of ints
        RGB values of final color
    
    N : int
    
        The number of values to interpolate
        
    Returns
    -------
    
    col_list : list of tuples
        An ordered list of colors for the colormap
    
    '''
    
    col1 = array([item/255. for item in col1])
    col2 = array([item/255. for item in col2])
    
    vr = list()
    for ii in range(3):
        vr.append(linspace(col1[ii],col2[ii],N))
    colist = array(vr).T
    return [tuple(thing) for thing in colist]




def overlay_images(dir1, dir2, out_dir, ftype1='.png',ftype2='.png'):
    '''
    Given two directories full of images, load one image from each directory and 
    overlay it on the other with the specified tint and color

    Parameters
    ----------

    dir1 : str
        Path to the directory of images that will be used as the background

    dir2 : str
        Path to the directory of images that will be tinted and overlaid

    out_dir : str
        Path to the directory at which output will be saved

    '''
  

    bg_ims = glob.glob(dir1+'/*'+ftype1)
    fg_ims = glob.glob(dir2+'/*'+ftype2)
    bg_color = (153/255., 204/255., 255/255.)
    fg_color = (255/255., 204/255., 102/255.)

    if len(bg_ims) != len(bg_ims):
        warnings.warn("The two image directories contain different numbers of images.")
    
    for ind, bg_im in enumerate(bg_ims):
        
        fg_im = fg_ims[ind]
        im1 = imread2(bg_im)
        if len(im1.shape)==3:
            im1 = sum(im1, axis=2)/3.

        im2 = imread2(fg_im)
        if len(im2.shape)==3:
            im2 = sum(im2, axis=2)/3.


        im2_norm = im2.astype(double)/255.
        im2_mask = im2_norm < .2
        
        im2_norm[im2_mask] = 0.0
        just_bigger_sizes = im2_norm*im1
        just_bigger_sizes = grey_dilation(just_bigger_sizes, size=(2,2))
        
        im2_norm = im2.astype(double)/255.
#         im2_mask = im2_norm > .2
        im2_norm[~im2_mask] = 0.0
        just_smaller_sizes = im2_norm*im1
        
        just_smaller_sizes = grey_dilation(just_smaller_sizes, size=(2,2))
        
        if ind==0:
            norm_factor1 = max(ravel(just_smaller_sizes)) 
            norm_factor2 = max(ravel(just_bigger_sizes)) 
        just_smaller_sizes = (just_smaller_sizes.astype(double)/norm_factor1)*255
        just_bigger_sizes = (just_bigger_sizes.astype(double)/norm_factor2)*255
        
        
        rgb_bg = concatenate([(just_smaller_sizes*chan)[...,newaxis] for chan in bg_color],axis=-1)
        
        rgb_img = concatenate([(just_bigger_sizes*chan)[...,newaxis] for chan in fg_color],axis=-1)
        
        finim = rgb_bg + rgb_img

        bg_name = os.path.split(bg_im)[-1][:-4]
        fg_name = os.path.split(fg_im)[-1][:-4]
        savestr = out_dir +'/'+bg_name+'_times_'+fg_name+'.png'

        if ind==0:
            cmax0=max(ravel(finim))
        toimage(finim, cmin=0.0, cmax=cmax0).save(savestr)
