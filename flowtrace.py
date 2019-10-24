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
import numpy as np
import warnings
import os
import glob

import math


from skimage.io import imsave
from skimage.io import imread as imread2
from PIL import Image



try:
    from scipy.ndimage.morphology import grey_dilation
except ImportError:
    warnings.warn('scipy.ndimage.morphology not imported, this may be required in future versions')


try:
    import multiprocessing as mp
    parallel_available = True
except ImportError:
    parallel_available = False
    warnings.warn('No multiprocessing module found, running code on single processor only')





_errstr = "Mode is unknown or incompatible with input array shape."

## Copy of a Deprecated Scipy Function
def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

## Copy of a Deprecated Scipy Function
def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image


def flowtrace(image_dir, frames_to_merge, out_dir='same', use_parallel=True, max_cores=8, frames_to_skip=1, **kwargs):
    '''
    This code is optimized for computers that DO NOT have enough 
    RAM to store the entire time series in memory, but which DO have enough 
    RAM to store an entire z-projection stack in memory. When using multithreading,
    be sure to multiply the memory usage per stack by the number of cores that will be used.
    
    Args:
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
    
    Keyword Args:
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
    '''
    image_files = glob.glob(image_dir+'/*.tif')
    image_files.sort()
    image_files = image_files[::frames_to_skip]
    im_path = os.path.split(image_files[0])[:-1][0]
    if out_dir == 'same':
        out_dir = image_dir
    
    
    if use_parallel and parallel_available:

        # Set up a list of processes that we want to run
        num_cores = mp.cpu_count()
        
        num_cores = min([num_cores, max_cores])

        nframes = len(image_files)

        runlen = math.ceil(nframes/num_cores) # think about whether this will throw an error and floor is safer
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
    
    Args:
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
        frame0 = np.sum(frame0, axis=2)/3.
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
                stack = np.roll(stack, -1, axis=3)
            else:
                stack = np.roll(stack, -1, axis=2)
            im = imread2(image_files[ii+frames_to_merge])
            stack[...,-1] = im
        stack2 = stack.copy()


        if 'subtract_first' in kwargs:
            if kwargs['subtract_first']:
                front_im = stack2[...,0]
                stack2 = stack2-front_im[..., np.newaxis]
        
        if 'subtract_median' in kwargs:
            if kwargs['subtract_median']:
                med_im= median(stack2, axis=-1)
                stack2 = stack2-med_im[..., np.newaxis]
        
        if 'take_diff' in kwargs:
            if kwargs['take_diff']:
                stack2 = diff(stack2, n=diff_order, axis=-1)

        if 'add_first_frame' in kwargs:        
            if kwargs['add_first_frame']:
                # stack2 = dstack([stack2, stack[...,0]])
                stack2 = np.concatenate((stack2, np.expand_dims(stack[..., 0], axis=-1)), axis=-1)
        
        if 'color_series' in kwargs:
            if kwargs['color_series']:

                # probably want to preallocate this upstream (255, 204, 153)
                # fullcmap = array(cmap1D((0,250,154),(255,20,147), stack2.shape[-1]))/255.
                fullcmap = np.array(cmap1D((90, 10, 250),(255, 153, 0),stack2.shape[-1]))/255.

                rvals, gvals, bvals = stack2*fullcmap[:, 0], stack2*fullcmap[:, 1], stack2*fullcmap[:, 2]
                stack2 = np.concatenate([rvals[..., np.newaxis], gvals[..., np.newaxis], bvals[..., np.newaxis]], axis=-1)
                stack2 = np.swapaxes(stack2,-1,-2)

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

        savestr = out_dir + '/' + im_name+  '_streamlines' + '_frames' + str(frames_to_merge) + '.png'

        # This makes it difficult to save with a light background, and it also might cause flickering
        # due to re-normalization
        toimage(max_proj, cmin=0.0, cmax=np.max(np.ravel(max_proj))).save(savestr)


def overlay_images(dir1, dir2, out_dir, ftype1='.tif', ftype2='.tif', 
    tint_color = (127/255., 255/255., 212/255.)):
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
    tint_color : 3-list
        Three values between 0 and 1 specifying a color in RGB

    '''  

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
    
    col1 = np.array([item/255. for item in col1])
    col2 = np.array([item/255. for item in col2])
    
    vr = list()
    for ii in range(3):
        vr.append(np.linspace(col1[ii],col2[ii],N))
    colist = np.array(vr).T
    return [tuple(thing) for thing in colist]




def overlay_images(dir1, dir2, out_dir, ftype1='.png',ftype2='.png',
    bg_color=(153/255., 204/255., 255/255.), fg_color=(255/255., 204/255., 102/255.)):
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
    bg_color : 3-list
        Three values between 0 and 1 specifying a color in RGB
    fg_color : 3-list
        Three values between 0 and 1 specifying a color in RGB

    '''
  

    bg_ims = glob.glob(dir1+'/*'+ftype1).sort()
    fg_ims = glob.glob(dir2+'/*'+ftype2).sort()
    
    

    if len(bg_ims) != len(fg_ims):
        warnings.warn("The two image directories contain different numbers of images.")
    
    for ind, bg_im in enumerate(bg_ims):

        fg_im = fg_ims[ind]
        im1 = imread2(bg_im)
        if len(im1.shape)==3:
            im1 = np.sum(im1, axis=2)/3.

        im2 = imread2(fg_im)
        if len(im2.shape)==3:
            im2 = np.sum(im2, axis=2)/3.

        im2_norm = im2.astype(double)/255.
        im2_mask = im2_norm < .2
        
        im2_norm[im2_mask] = 0.0
        just_bigger_sizes = im2_norm*im1
        just_bigger_sizes = grey_dilation(just_bigger_sizes, size=(2,2))
        
        im2_norm = im2.astype(double)/255.
        # im2_mask = im2_norm > .2
        im2_norm[~im2_mask] = 0.0
        just_smaller_sizes = im2_norm*im1
        
        just_smaller_sizes = grey_dilation(just_smaller_sizes, size=(2,2))
        
        if ind==0:
            norm_factor1 = np.max(np.ravel(just_smaller_sizes)) 
            norm_factor2 = np.max(np.ravel(just_bigger_sizes)) 
        just_smaller_sizes = (just_smaller_sizes.astype(double)/norm_factor1)*255
        just_bigger_sizes = (just_bigger_sizes.astype(double)/norm_factor2)*255
        
        rgb_bg = np.concatenate([(just_smaller_sizes*chan)[..., np.newaxis] for chan in bg_color], axis=-1)
        rgb_img = np.concatenate([(just_bigger_sizes*chan)[..., np.newaxis] for chan in fg_color], axis=-1)
        finim = rgb_bg + rgb_img

        bg_name = os.path.split(bg_im)[-1][:-4]
        fg_name = os.path.split(fg_im)[-1][:-4]
        savestr = out_dir +'/'+bg_name+'_times_'+fg_name+'.png'

        if ind==0:
            cmax0=np.max(np.ravel(finim))
        toimage(finim, cmin=0.0, cmax=cmax0).save(savestr)
