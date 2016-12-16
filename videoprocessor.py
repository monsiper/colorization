def image_maker(video_name):
    """
    Create list of images from video, and store the images 
    in the current directory.
    video_name name of the video 
    """
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    count = 0
    success = True
    L=[]
    while success:
        success,image = vidcap.read()
        #print 'Read a new frame: ', success
        L.append('frame'+str(count)+'.jpg')
        cv2.imwrite("frame%d.jpg" % count, image)     
        count += 1
        
def video_maker(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID"):
    """
    Create a video from a list of images.
    outvid      output video
    images      list of images to use in the video
    fps         frame per second
    size        size of each frame
    is_color    color
    format      
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    outvid='result.mp4'
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid
    

    
    