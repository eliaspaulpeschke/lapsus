#!/usr/bin/env python3
import numpy as np
import importlib.util
import cv2 as cv
import time
import argparse
import math
import os

verbose = False
debug = False
log = False
savenumber = 0
force_dummy = False

def capture(cam):
    cam.start()
    time.sleep(0.2)
    img = cam.capture_image()
    cam.stop()
    return toCv(img)

def toCv(pil_img):
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

def thresh(im, blur=True, blur_size=(5,5)):
    g = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(g,blur_size,0)
    ret,th = cv.threshold(blur,0,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C+cv.THRESH_OTSU)
    return th

def compare_xor(a, b):
    assert a.shape == b.shape
    ta = thresh(a)
    tb = thresh(b)
    diff = cv.bitwise_xor(ta, tb)
    norm = diff.sum() / (a.shape[0] * a.shape[1])
    return (norm, diff)

def compare_changed_area(a, b, min_area, area_thresh, blur_before = (25,25), blur_thresh=(25,25), tozero=15):
    a_blur = cv.GaussianBlur(a,blur_before,0)
    b_blur = cv.GaussianBlur(b,blur_before,0)
    diff = cv.absdiff(a_blur, b_blur)
    diff = cv.threshold(diff,tozero ,255,cv.THRESH_TOZERO)[1]
    diff_thresh = 255 - thresh(diff, blur=True, blur_size=blur_thresh)
    contours, _ = cv.findContours(diff_thresh, 1, 2)
    big_enough = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
    big_areas = [cv.contourArea(cnt) for cnt in big_enough]
    #big_areas = [cv.contourArea(cnt) for cnt in contours if cv.contourArea(cnt) > min_area]
    area_sum = np.sum(big_areas)
    #cont_source = a.copy()
    #cont_img = cv.drawContours(cont_source, big_enough, -1, (0,0,255), 4)
    return area_sum >= area_thresh#, cont_img


def show(*args, size=(1024,768)):
    i = 0
    for img in args:
        cv.imshow(f"image{i}", cv.resize(img,size))
        i += i
    cv.waitKey(0)
    cv.destroyAllWindows()

def handle_args():
    global verbose, debug, log, force_dummy
    parser = argparse.ArgumentParser(prog='lapsus',
            description='adaptive timelapse utility for the raspberry pi', epilog='BOTTOM TEXT')
    parser.add_argument("--base-interval", type=float, default=1.0, 
            help="base interval in which to capture an image and test if it is"\
                    "different than the last one, in seconds.")
    parser.add_argument("--max-interval", type=float, default = 0, 
            help="capture an image after this interval in seconds,"\
                    "even if it is no different than before. if set to zero or not set,"\
                    "only capture on change")
    parser.add_argument("--change", type=float, default=1000.0, 
            help="minimum change to justify saving an image."\
                    "area of all changed areas big enough to count added together")
    parser.add_argument("--change-chunksize", type=float, default=1000.0, 
            help="how big must a changed area be to cout as changes?")
    parser.add_argument("--directory", type=str, default="./", 
            help="where to save images and log")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--force-dummycam", action="store_true", help="force using a dummy camera - for debugging")
    parser.add_argument("--log", action="store_true", help="logs absolute thresholded difference for each image if set")
    args = parser.parse_args()
    verbose = args.verbose
    debug = args.debug
    log = args.log
    force_dummy = args.force_dummycam
    return args

def get_cam():
    if (not force_dummy) and (importlib.util.find_spec("picamera2") is not None):
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure("still")
        return cam
    elif debug or force_dummy:
        print("WARNING! Picamera2 was not found. Using a dummy camera that will only capture black images")
        from PIL import Image
        class Cam():
            def __init__(self):
                pass
            def capture_image(self):
                return Image.new('RGB', (100,100))
            def start(self):
                pass
            def stop(self):
                pass
        return Cam()
    else:
        raise ModuleNotFoundError("Could not find picamera2. Try debug flag for testing without a camera")

def main():
    global logfile
    args = handle_args()
    if verbose:
        print(args.change, args.max_interval, args.base_interval)
    if not(os.path.exists(args.directory)):
        os.makedirs(args.directory)
    if log:
        logfile = os.path.join(args.directory, ("lapsus-" + str(time.strftime('%d%m%y-%H%M%S') + ".log")))
        with open(logfile, "a") as logf:
            logf.write("Starting lapsus now")
    save_dir = os.path.join(args.directory, f"{time.strftime('%d%m%y-%H%M%S')}")
    os.mkdir(save_dir)
    cam = get_cam()
    def compare(a, b):
        return compare_changed_area(a, b, args.change_chunksize, args.change)
    mainloop(compare, cam, save_dir, args)
  
def mainloop(compare, cam, save_dir, args):
    last_image = capture(cam)
    last_capture = time.time()
    last_save = last_capture
    save_image(last_image, save_dir)
    while True:
        start_time = time.time()
        if (start_time - last_capture) > args.base_interval:
            cap = capture(cam)
            last_capture = start_time
            if verbose:
                print(f"Captured. Seconds since last save {last_capture - last_save}.")
            if compare(last_image, cap) or ((last_capture - last_save) > args.max_interval and args.max_interval != 0):
                last_save = last_capture
                norm, _ = compare_xor(last_image, cap)
                save_image(cap, save_dir, f" mean difference: {norm}")
                last_image = cap


def save_image(im, directory, additional_info = ""):
    global savenumber
    filename = os.path.join(directory, f"{savenumber}-{str(time.time()).replace('.','-')}.jpg")
    savenumber += 1
    if verbose:
        print("WRITING:", filename)
    if log:
        with open(logfile, "a") as logf:
            logf.write(filename + " " + additional_info + "\n")
    cv.imwrite(filename,im)

if __name__ == "__main__":
    main()
