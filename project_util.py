import urllib2
from bs4 import BeautifulSoup
import os, sys
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from skimage import color, io,measure
from theano.tensor.signal import pool
import theano
from sklearn.neighbors import NearestNeighbors

def shared_dataset(data, borrow=True):

    shared_data = theano.shared(np.asarray(data,dtype=theano.config.floatX),
                                borrow=borrow)

    return shared_data

def download_images(dir_name, num_of_pages):
    """Function that scrapes the images from www.themovie.db and saves them as .jpeg into
    'folder_contains_this_file/data/' folder"""

    new_path = os.path.join(
        os.path.split(__file__)[0], dir_name+'/raw')

    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    url_list = []

    for page in range(num_of_pages):
        if page==0:
            res = urllib2.urlopen( 'https://www.themoviedb.org/movie/top-rated' )
        else:
            res = urllib2.urlopen( 'https://www.themoviedb.org/movie/top-rated?page=%s'%page )

        soup = BeautifulSoup(res.read())
        links = soup.find_all('a',{'class':'title result'} )

        # for link in links:
        for link in links:
            url = 'https://www.themoviedb.org' + link.get('href') + '-' + (link.get('title').lower()).replace(" ","-")+'/images/backdrops'
            url_list.append(url)

    print('Downloading images...')
    for url in url_list:
        try:
            res = urllib2.urlopen(url)
            soup = BeautifulSoup(res.read())
            images = soup.find_all('img',{'class':'backdrop lazyload'})
            for image in images:
                full_filename = image.get('data-src')
                filename = str(full_filename.split('/')[-1])
                data = urllib2.urlopen(full_filename).read()
                complete_path = os.path.join(new_path,filename)
                with open(complete_path, "wb") as code:
                    code.write(data)
        except Exception:
            pass

def prepare_image_sets(dir_name='data', batch_size=10000, dataset_type='training', model_type='prob', threshold=5):
    """Converts .jpeg images into numpy matrices(each row corresponds to one image) and saves
     them in batches consisting of batch_size images"""

    if dataset_type=='training':

        path = os.path.join(
            os.path.split(__file__)[0], dir_name+'/raw')
        new_path = os.path.join(
            os.path.split(__file__)[0], dir_name)

        if not os.path.isdir(path):
            return "Data directory doesn't exist"

        l_out, ab_enc_out = 0, 0
        batch_reset = True
        batch_ind = 0
        img_ind = 0
        for image in os.listdir(path):
            if not image==".DS_Store" and not os.path.splitext(image)=='.npy':
                with open(path + '/' + image, 'r+b') as f:
                    with Image.open(f) as img_f:
                        try:
                            resized_im = resizeimage.resize_cover(img_f, [256, 256])
                            img_rgb = (np.array(resized_im))
                            if len(img_rgb.shape)==3:
                                img_lab = color.rgb2lab(img_rgb[:,:,0:3])
                                #store lab space differently depending on model_type
                                if model_type=='prob':
                                    l = img_lab[:, :, 0].flatten() # Slicing to get L data
                                    a = (img_lab[:, :, 1][::4,::4]).flatten()  # Slicing to get a data
                                    b = (img_lab[:, :, 2][::4,::4]).flatten()  # Slicing to get b data
                                if model_type=='regression':
                                    l = img_lab[:, :, 0].flatten()  # Slicing to get L data
                                    a = img_lab[:, :, 1].flatten()  # Slicing to get a data
                                    b = img_lab[:, :, 2].flatten()

                                #print(np.sum(a>threshold)+np.sum(b>threshold))
                                if (np.sum(a>threshold)+np.sum(b>threshold))>1:
                                    if batch_reset:
                                        l_out = np.array(l, np.uint8)
                                        ab_enc_out = encode_ab_to_Q(a,b,'regression')
                                        batch_reset = False
                                    else:
                                        new_l = l.astype(dtype=np.uint8)
                                        new_ab_enc = encode_ab_to_Q(a,b,'regression')
                                        l_out = np.vstack((l_out,new_l))
                                        ab_enc_out = np.vstack((ab_enc_out, new_ab_enc))
                                    img_ind += 1

                                    if img_ind%batch_size==0 and img_ind!=0:
                                        batch_ind += 1
                                        print('Creating matrices for batch number %s' %batch_ind)
                                        np.save(new_path + '/test_batch_l_%s.npy' %batch_ind, l_out)
                                        np.save(new_path + '/test_batch_ab_%s.npy' % batch_ind, ab_enc_out)
                                        batch_reset = True
                                else:
                                    print('grayscale found: %s' %image)
                        except:
                            print('error image: %s' %image)
            if batch_ind==50:
                break

        if not batch_reset:
            batch_ind += 1
            np.save(new_path + '/test_batch_l_%s.npy' % batch_ind, l_out)
            np.save(new_path + '/test_batch_ab_%s.npy' % batch_ind, ab_enc_out)

    elif dataset_type=='frames':
        path = ('/Users/monsiper/Downloads/frames/')
        new_path = ('/Users/monsiper/Downloads/frames/processed/')

        l_out = 0
        batch_reset = True
        batch_ind = 0
        img_ind = 0
        for image in os.listdir(path):
            if not image == ".DS_Store" and not image == 'processed':
                with open(path + image, 'r+b') as f:
                    with Image.open(f) as img_f:
                        resized_im = resizeimage.resize_cover(img_f, [256, 256])
                        # resized_im.save(path + '/' + 'image-%s.jpeg' % index, img_f.format)
                        img_rgb = (np.array(resized_im))
                        if len(img_rgb.shape) == 3:
                            img_ind += 1
                            img_lab = color.rgb2lab(img_rgb[:, :, 0:3])
                            l = img_lab[:, :, 0].flatten()  # Slicing to get L data

                            if batch_reset:
                                l_out = np.array(l, np.float32)
                                batch_reset = False
                            else:
                                new_l = l.astype(dtype=np.float32)
                                l_out = np.vstack((l_out, new_l))

                            if img_ind % batch_size == 0 and img_ind != 0:
                                batch_ind += 1
                                print('Creating matrices for batch number %s' % batch_ind)
                                np.save(new_path + '/frame_batch_l_%s.npy' % batch_ind, l_out)
                                batch_reset = True

        if not batch_reset:
            batch_ind += 1
            np.save(new_path + '/frame_batch_l_%s.npy' % batch_ind, l_out)

def load_data(dir_name, theano_shared=True, ds=1,batch_ind=None,batch_num=1,load_frames=False,shuffler=True):
    #load data into theano shared variables
    path = os.path.join(
        os.path.split(__file__)[0], dir_name)

    if not os.path.isdir(path):
        return "Data folder doesn't exist"

    train_batches = []
    for file in os.listdir(path):
        if os.path.splitext(file)[-1]=='.npy':
            train_batches.append(file)

    if not train_batches:
        return 'There is no .npy file in data folder'
    if load_frames:
        print('Loading Frame Batch %s'%(batch_ind))
        train_set_l = np.load(path + '/frame_batch_l_%s.npy'%(batch_ind))
        for i in range(batch_ind,batch_ind+batch_num-1):
            new_set_l = np.load(path + '/frame_batch_l_%s.npy'%(i+1))
            train_set_l = np.concatenate((train_set_l, new_set_l), axis=0)
        train_set_l.astype(theano.config.floatX)
        if theano_shared:
            train_set_l_mat = shared_dataset(train_set_l)
        else:
            train_set_l_mat = train_set_l
        return train_set_l_mat
    
    else:
        print('Loading Batch %s'%(batch_ind))
        train_set_l = np.load(path + '/test_batch_l_%s.npy'%(batch_ind))
        train_set_ab = np.load(path + '/test_batch_ab_%s.npy'%(batch_ind))
        for i in range(batch_ind,batch_ind+batch_num-1):
            print('Loading Batch %s'%(i+1))
            new_set_l = np.load(path + '/test_batch_l_%s.npy'%(i+1))
            new_set_ab = np.load(path + '/test_batch_ab_%s.npy'%(i+1))
            train_set_l = np.concatenate((train_set_l, new_set_l), axis=0)
            train_set_ab = np.concatenate((train_set_ab, new_set_ab), axis=0)
        if shuffler:
            np.random.seed(35)
            np.random.shuffle(train_set_l)
            np.random.seed(35)
            np.random.shuffle(train_set_ab)
        train_set_l.astype(theano.config.floatX)
        train_set_ab.astype(theano.config.floatX)
        if theano_shared:
            train_set_l_mat = shared_dataset(train_set_l)
            train_set_ab_mat = shared_dataset(train_set_ab)
        else:
            train_set_l_mat = train_set_l
            train_set_ab_mat = train_set_ab

        return (train_set_l_mat, train_set_ab_mat)

def encode_ab_to_Q(a_chan_flt, b_chan_flt, model_type):
    #soft encode ab space to Q color bins
    if model_type =='prob':
        sigma = 5.
        ab_chan_comb = np.column_stack((a_chan_flt,b_chan_flt))
        ref_Qcolor_bins = np.load('pts_in_hull.npy')
        img_enc = np.zeros((a_chan_flt.shape[0], ref_Qcolor_bins.shape[0] ))
        x_ind =np.arange(0,a_chan_flt.shape[0],dtype='int')[:,np.newaxis]
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(ref_Qcolor_bins)
        distances, indices = nbrs.kneighbors(ab_chan_comb)
        wts = np.exp(-distances**2/(2*sigma**2))
        wts = wts/np.sum(wts, axis=1)[:, np.newaxis]
        img_enc[x_ind, indices] = wts

        return (img_enc.flatten()).astype(dtype=np.float32)

    if model_type=='regression':
        return np.array(list(a_chan_flt) + list(b_chan_flt), dtype=np.float32)





