def image_maker(video_name):
    """
    Create list of images from video, and store the images 
    in the current directory.
    video_name name of the video 
    """
    vidcap = cv2.VideoCapture(video_name)
    success,image = vidcap.read()
    success = True
    L=[]
    while success:
        success,image = vidcap.read()
        L.append(image)
    return L 
        
def video_maker(images, outimg=None, fps=25, size=None,
               is_color=True, format="MJPG"):
    """
    Create a video from a list of images.
    outvid      output video
    images      list of images to use in the video
    fps         frame per second
    size        size of each frame
    is_color    color
    format      four character code of the format. 
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    outvid='result.avi'
    for image in images:
        img = image
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        try: 
            if size[0] != img.shape[1] and size[1] != img.shape[0]:
                img = resize(img, size)
            vid.write(img)
        except:
            pass 
    vid.release()
    return vid
L=image_maker('test_video.mp4')
print ("extracted frames, now creating video......")
video_maker(L, outimg=None, fps=25, size=None,
               is_color=True, format="MJPG")

    

    
    