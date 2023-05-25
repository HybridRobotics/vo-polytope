import os
from pathlib import Path
import cv2 as cv

image_path = Path(__file__).parent / 'image'
media_path = Path(__file__).parent / 'media'

if media_path.exists():
    pass
else:
    media_path.mkdir()
    print('Get!')


def image_to_video(image_pa, media_pa):
    """ Compose pictures into videos """

    # get all names of pictures in the image_path
    image_names = os.listdir(image_pa)

    # sort
    image_names.sort(key=lambda n: int(n[:-4]))

    # set the format
    fourcc = cv.VideoWriter_fourcc('M', 'P', '4', 'V')

    # set the frame
    fps = 5

    # Get the size of the first figure
    img = cv.imread(str(image_pa) + '/' + image_names[0])
    imgInfo = img.shape

    # initialize the media target
    media_writer = cv.VideoWriter(str(media_path) + '/polytope_vo.mp4', fourcc, fps, (int(imgInfo[1]), int(imgInfo[0])))
    for image_name in image_names:
        im = cv.imread(os.path.join(image_pa, image_name))
        media_writer.write(im)
        print(image_name, 'OK！')

    # release the media target
    media_writer.release()


if __name__ == "__main__":
    image_to_video(image_path, media_path)

