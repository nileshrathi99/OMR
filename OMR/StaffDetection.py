from PIL import ImageDraw
import numpy as np


def preprocess_image(image):
    return image.convert('L')


''' get all the potential lines which are at 0 degree '''
def get_lines(im):

    accumulator = np.zeros(im.height, dtype=np.int16)
    im_ = np.array(im)

    for x in range(im_.shape[0]):
        whites = 0
        blacks = 0
        for y in range(im_.shape[1]):
            if im_[x][y] < 200:
                blacks += 1
            else:
                whites +=1
        if(blacks > whites):
            accumulator[x] += 1

    lines_detected = []

    for x in range(im_.shape[0]-1):
        if accumulator[x] == 1 and accumulator[x] !=  accumulator[x+1]:
            lines_detected.append(x)

    lines_detected_ = []
    lines_detected_.append(lines_detected[0])
    for i in range(1, len(sorted(lines_detected))):
        if abs(lines_detected[i] - lines_detected[i-1]) < 8:
            lines_detected_.append(lines_detected[i-1])
        else:
            lines_detected_.append(lines_detected[i])

    lines_detected_ = list(set(lines_detected_))

    return lines_detected_


''' draw the detected lines over the image '''
def draw_lines(lines_detected_, img):

    if img.mode == 'L':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)

    for i in range(len(lines_detected_)):
        x1 = 0
        y1 = lines_detected_[i]
        x2 = img.width
        y2 = y1
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)

    img.save('detectedLines.png')


''' return the initial postions of all the treble and base '''
def get_staff_positions(lines):
    
    lines = sorted(lines)
    treble_pos = []
    base_pos = []
    flag_trebble = False
    treble_pos.append(lines[0])
    for i in range(1, len(lines)):
        if lines[i] - lines[i-1] > 50:
            if flag_trebble:
                treble_pos.append(lines[i])
                flag_trebble = False
            else:
                base_pos.append(lines[i])
                flag_trebble = True
    
    return treble_pos, base_pos


def get_space(lines):
    sorted_lines = sorted(lines)[0:4]
    differences = [sorted_lines[i+1] - sorted_lines[i] for i in range(len(sorted_lines) - 1)]
    return int(sum(differences)/len(differences))


''' return the initial postions of all the treble and base and image '''
def get_treble_base(image):
    im = preprocess_image(image)
    lines_detected_ = get_lines(im)
    print(lines_detected_)
    # draw_lines(lines_detected_, image)
    treble_pos_list, base_pos_list =  get_staff_positions(lines_detected_)
    space =  get_space(lines_detected_)
    return treble_pos_list, base_pos_list, space




