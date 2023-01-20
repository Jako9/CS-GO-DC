from argparse import ArgumentError
import cv2
import numpy as np
import os
import time
import pandas as pd
import json
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# List of characters to check in key_check()-function (see key_input.py)
# x11 = CTRL, 0x20 = SPACE
KEYLIST = ['W', 'A', 'S', 'D','\x11', '\x20', 'R']


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,1)

def thresholding(image):
    #return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # (bild, threshold, upperBound, thresholding-type)
    return cv2.threshold(image, 225, 255, cv2.THRESH_BINARY_INV)[1]

def rescale(img, scale=1):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def display_screenshot(screenshot):
    cv2.imshow('', screenshot)
    cv2.waitKey(0) # wait for user to close window
    cv2.destroyAllWindows()
    
    
def inspect_recording(path, fps=5):
    stepsize = 20
    session_name = os.path.split(path)[-1]
    
    arr = np.load(os.path.join(path, session_name + '_screenshot.npy'))
    brr = np.load(os.path.join(path, session_name + '_showpos_infos.npy'))
    dataframe = pd.read_pickle(os.path.join(path, session_name + '_user_actions.pkl'))
    print(dataframe)
    
    for idx in range(0, arr.shape[0], stepsize):
        print(dataframe.iloc[idx])
        display_screenshot(arr[idx])
        img = get_grayscale(brr[idx])
        img = thresholding(img)
        print(np.max(img))
        print(img.shape)
        print(img2str(img, preprocess=0))
        display_screenshot(img)
   

def img2str(img, preprocess=1, debug=False):
    if preprocess:
        img = get_grayscale(img)
        img = thresholding(img)
    
    # Page segmentation mode (psm): 3 (default), 5 oder 6 probieren
    # OCR Engine Mode (oem): 
        #0    Legacy engine only.
        #1    Neural nets LSTM engine only.
        #2    Legacy + LSTM engines.
        #3    Default, based on what is available.
    # tessedit_char_blacklist=0123456789    
    
    custom_config = r'-c tessedit_char_whitelist="-_ 0123456789." --psm 7 --oem 3'
    text = pytesseract.image_to_string(img, config=custom_config)
    #data = pytesseract.image_to_data(img, config=custom_config)
    #text = pytesseract.image_to_boxes(img, config=custom_config)
    
    if False:
        print('-'*50)
        print(text)
        print(data)
        display_screenshot(img)
    
    return text
   

def get_pos_info_from_str(text):
    pos_flag = 1
    ang_flag = 1
    vel_flag = 1
    
    # Splitting read text row-wise
    THRESHOLD = 3 # maybe bigger?
    text = [x for x in text.split('\n') if len(x) > THRESHOLD]
    
    # Regex
    float_regex = r'-?\d+.\d\d'
    pos_float_regex = r'\d+.\d\d'
    
    # Read position information
    try:
        pos_text = text[0]
        pos_coords = [parse_number(x) for x in re.findall(float_regex, pos_text)]    
        assert len(pos_coords) == 3
    # AttributeError:   re.findall None
    # AssertionError:   
    # IndexError:        text leer -> pos_text nicht existent
    except (AttributeError, AssertionError, IndexError):
        try:
            pos_text = pos_text.split('\s')
            assert len(pos_text) == 3
            pos_coords = [parse_number(pos_text[i]) for i in [0,1,2]]
        except:
            pos_coords = [0,0,0]
            pos_flag = 0
        
    # Read angle information
    try:
        ang_text = text[1]
        ang_coords = [parse_number(x) for x in re.findall(float_regex, ang_text)]
        assert len(ang_coords) == 3
        ang_coords[2] = 0    
    except (AttributeError, AssertionError, IndexError):
        try:
            ang_coords = ang_text.split('\s')
            assert len(ang_text) == 3
            ang_coords = [parse_number(ang_text[i]) for i in [0,1,2]]
            ang_coords[2] = 0
        except:
            ang_coords = [0,0,0]
            ang_flag = 0
    
    # Read velocity information 
    try:
        vel_text = text[2]
        vel_text = re.match(pos_float_regex, vel_text).group()
        vel = parse_number(vel_text)
    except (AttributeError, AssertionError, IndexError):
        vel = 0
        vel_flag = 0
    
    return [pos_coords, pos_flag, ang_coords, ang_flag, vel, vel_flag]


def parse_number(x):
    try:
        return float(x)
    except ValueError:
        res = ''
        for idx, char in enumerate(x):
            if idx > 0 and (char in ['-', ' ']):
                res += '.'
            else:
                res += char
        try: 
            return float(res)
        except ValueError:
            print('Cant deal with {}'.format(res))
            return -12345


def get_pos_info_from_str_depr(text):
    float_regex = r'-?\d+.\d\d'
    pos_float_regex = r'\d+.\d\d'
    place_holder_zeros = r'0.00'
    pos_regex = r'pos:\s*{}\s*{}\s*{}'.format(float_regex, float_regex, pos_float_regex)
    ang_regex = r'ang:\s*{}\s*{}\s*{}'.format(float_regex, float_regex, place_holder_zeros)
    vel_regex = r'vel:\s*{}'.format(pos_float_regex)
        
    try:
        pos_text = re.search(pos_regex, text).group()
        ang_text = re.search(ang_regex, text).group()
        vel_text = re.search(vel_regex, text).group()
    except AttributeError:
        try:
                        
            pos_text = text.split('\n')[0]
            ang_text = text.split('\n')[1]
            vel_text = text.split('\n')[2]
        except:
            return False
        
    try:    
        pos_coords = [float(x) for x in re.findall(float_regex, pos_text)]
        assert len(pos_coords) == 3
    except (ValueError, AssertionError):
        try:
            pos_text = pos_text.split('\s')
            pos_coords = [float(pos_text[i]) for i in [1,2,3]]
        except:
            return False
        
    try:
        ang_coords = [float(x) for x in re.findall(float_regex, ang_text)]
        assert len(ang_coords) == 3
    except (ValueError, AssertionError):
        try:
            ang_text = ang_text.split('\s')
            ang_coords = [float(ang_text[i]) for i in [1,2,3]]
        except:
            return False
        
    try:
        vel_value = re.search(float_regex, vel_text).group()
        vel = float(vel_value)
    except AttributeError:
        try:
            vel_text = vel_text.split('\s')
            vel = float(vel_text[1])
        except:
            return False
    
    return [pos_coords, ang_coords, vel]
    

def create_dir(path, name):
    out_dir = os.path.join(path, name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir


def init_recording_session(path):
    if os.path.exists(path):
        session_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        return create_dir(path, session_name)
    else:
        print('{} does not exist!'.format(path))
        exit(-1)
        
        

def save_recording_session(path, screenshots, keys, mouse_states, meta_data=None):
    # Retreiving session name from directory-path
    session_name = os.path.split(path)[-1]
    
    # Creating np.arrays
    screenshots = np.array(screenshots, dtype=np.uint8)
    print('Shape of screenshots {}'.format(screenshots.shape))
    mouse_states = np.array(mouse_states, dtype=np.int16)
    print('Shape of mouse_states = {}'.format(mouse_states.shape))
    
    # Saving screenshots
    f1 = os.path.join(path, session_name + '_screenshot.npy')
    np.save(f1, screenshots)
    print('Saving screenshots done.')
    
    
    # Saving player-actions in a dataframe
    d = {
        'W'     : [int('W' in k) for k in keys],
        'A'     : [int('A' in k) for k in keys],
        'S'     : [int('S' in k) for k in keys],
        'D'     : [int('D' in k) for k in keys],
        'CTRL'  : [int('\x11' in k) for k in keys],
        'SPACE' : [int('\x20' in k) for k in keys],
        'R'     : [int('R' in k) for k in keys],
        'Mouse_current_status' : mouse_states[:,0],
        'Mouse_clicked': mouse_states[:, 1],
        'Mouse_held_down': mouse_states[:, 2],
    }
    
    # Storing dataframe
    d = pd.DataFrame(data=d)
    f2 = os.path.join(path, session_name + '_user_actions.pkl')
    d.to_pickle(f2)
    print('Saving dataframe done.')
    
    
    # Storing metadata
    with open(os.path.join(path, session_name + '_metadata.json'), "w") as f:
        json.dump(meta_data, f, indent=4)
    print('Saving metadata done.')
    
    # Debugging
    print(d)
    
    #f2 = os.path.join(path, session_name + 'player_actions.npy')
    #np.save(f2, np.array())
    
    
if __name__ == '__main__':
    inspect_recording('E:\\test\\2022_04_23_23_27_07')
    
    #x = '-.15'
    #print(parse_number(x))