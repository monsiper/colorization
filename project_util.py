import urllib2
from bs4 import BeautifulSoup
import os, sys
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from matplotlib import pyplot as plt
from skimage import color, io
import theano

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

def prepare_image_sets(dir_name, batch_size=10000):
    """Converts .jpeg images into numpy matrices(each row corresponds to one image) and saves
     them in batches consisting of batch_size images"""

    path = os.path.join(
        os.path.split(__file__)[0], dir_name+'/raw')
    new_path = os.path.join(
        os.path.split(__file__)[0], dir_name)

    if not os.path.isdir(path):
        return "Data directory doesn't exist"

    l_out, ab_out = 0, 0
    batch_reset = True
    batch_ind = 0
    img_ind = 0
    for image in os.listdir(path):
        if not image==".DS_Store" and not os.path.splitext(image)=='.npy':
            with open(path + '/' + image, 'r+b') as f:
                with Image.open(f) as img_f:
                    resized_im = resizeimage.resize_cover(img_f, [256, 256])
                    # resized_im.save(path + '/' + 'image-%s.jpeg' % index, img_f.format)
                    img_rgb = (np.array(resized_im))
                    if len(img_rgb.shape)==3:
                        img_ind += 1
                        img_lab = color.rgb2lab(img_rgb)
                        l = img_lab[:, :, 0].flatten() # Slicing to get L data
                        a = img_lab[:, :, 1].flatten()  # Slicing to get a data
                        b = img_lab[:, :, 2].flatten()  # Slicing to get b data

                        if batch_reset:
                            l_out = np.array(l, np.float32)
                            ab_out = np.array(list(a) + list(b), np.float32)
                            batch_reset = False
                        else:
                            new_l = np.array(l, np.float32)
                            new_ab = np.array(list(a) + list(b), np.float32)
                            l_out = np.vstack((l_out,new_l))
                            ab_out = np.vstack((ab_out, new_ab))

                        if img_ind%batch_size==0 and img_ind!=0:
                            batch_ind += 1
                            print('Creating matrices for batch number %s' %batch_ind)
                            np.save(new_path + '/batch_l_%s.npy' %batch_ind, l_out)
                            np.save(new_path + '/batch_ab_%s.npy' % batch_ind, ab_out)
                            #np.save(new_path + '/batch_lab_%s.npy' % batch_ind,img_lab)
                            batch_reset = True

    if not batch_reset:
        batch_ind += 1
        np.save(new_path + '/batch_l_%s.npy' % batch_ind, l_out)
        np.save(new_path + '/batch_ab_%s.npy' % batch_ind, ab_out)

def load_data(dir_name, theano_shared=True,num_batch=1):

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

    train_set_l = np.load(path + '/batch_l_1.npy')
    train_set_ab = np.load(path + '/batch_ab_1.npy')


    for i in range(1,len(train_batches)/2):
        new_set_l = np.load(path + '/batch_l_%s.npy'%(i+1))
        new_set_ab = np.load(path + '/batch_ab_%s.npy'%(i+1))
        train_set_l = np.concatenate((train_set_l, new_set_l), axis=0)
        train_set_ab = np.concatenate((train_set_ab, new_set_ab), axis=0)

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

    return [train_set_l_mat, train_set_ab_mat]


def test_images():

    # download_images('data', 3)
    # prepare_image_sets('data',200)
    data_set = load_data('data', False)
    data_l = ((data_set[0][219]).reshape(1,256,256)).transpose(1,2,0)
    data_ab = ((data_set[1][219]).reshape(2,256,256)).transpose(1,2,0)
    img_construct = np.concatenate((data_l.astype(np.float64),data_ab.astype(np.float64)), axis=2)
    plt.imshow(color.lab2rgb(img_construct))
    plt.show()




# path = os.path.join(
#     os.path.split(__file__)[0], 'data')
#
# filename = path + '/' + '1AKL15pxyyVrHKMVR8Md64sAJw9.jpg'
# img_rgb = io.imread(filename)
# img_lab = color.rgb2lab(img_rgb)
# img_l = img_lab[:,:,0]
# img_ab = img_lab[:,:,1:4]
# img_construct = np.concatenate((img_l,img_ab), axis=2)
# plt.imshow(color.lab2rgb(img_construct))
# plt.show()


