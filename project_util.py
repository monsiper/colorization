import urllib2
from bs4 import BeautifulSoup
import os, sys
from PIL import Image
from resizeimage import resizeimage
import numpy as np
from matplotlib import pyplot as plt

def download_images(dir_name):
    """Function that scrapes the images from www.themovie.db and saves them as .jpeg into
    'folder_contains_this_file/data/' folder"""

    new_path = os.path.join(
        os.path.split(__file__)[0], dir_name)

    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    url_list = []

    for page in range(3):
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
        os.path.split(__file__)[0], dir_name)

    if not os.path.isdir(path):
        return "Data directory doesn't exist"

    out = 0
    batch_reset = True
    batch_ind = 0
    img_ind = 0
    for image in os.listdir(path):
        if not image==".DS_Store" and not os.path.splitext(image)=='.npy':
            with open(path + '/' + image, 'r+b') as f:
                with Image.open(f) as img_f:
                    resized_im = resizeimage.resize_cover(img_f, [256, 256])
                    # resized_im.save(path + '/' + 'image-%s.jpeg' % index, img_f.format)
                    img_mat = (np.array(resized_im))
                    if len(img_mat.shape)==3:
                        img_ind += 1
                        r = img_mat[:, :, 0].flatten() # Slicing to get R data
                        g = img_mat[:, :, 1].flatten()  # Slicing to get G data
                        b = img_mat[:, :, 2].flatten()  # Slicing to get B data

                        if batch_reset:
                            out = np.array(list(r) + list(g) + list(b), np.uint8)
                            batch_reset = False
                        else:
                            new_array = np.array(list(r) + list(g) + list(b), np.uint8)
                            out = np.vstack((out, new_array))

                        if img_ind%batch_size==0 and img_ind!=0:
                            batch_ind += 1
                            np.save(path + '/batch%s.npy' %batch_ind, out)
                            batch_reset = True

    batch_ind += 1
    np.save(path + '/batch%s.npy' % batch_ind, out)


def test_images(dir_name):
    path = os.path.join(os.path.split(__file__)[0], dir_name)

    for i in range(1, 4):
        data_mat = np.load(path + '/batch%s.npy' % 1)
        print data_mat.shape

    plt.imshow((data_mat[10].reshape(3, 256, 256)).transpose(1, 2, 0))
    plt.show()

test_images('./images/')


# path = os.path.join(
#     os.path.split(__file__)[0], 'data')

#download_images('data')
#prepare_image_sets('data')
