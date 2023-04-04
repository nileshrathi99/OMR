from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
import StaffDetection as staffDetection
import time 
from numba import jit

template_path_1 = 'template1.png'
template_path_2 = 'template2.png'
template_path_3 = 'template3.png'

treble_dict = {
                -2.0: 'B', 
                -1.5: 'A',
                -1.0: 'G',
                -0.5: 'F',
                   0: 'E',
                 0.5: 'D',
                 1.0: 'C',
                 1.5: 'B',
                 2.0: 'A',
                 2.5: 'G',
                 3.0: 'F',
                 3.5: 'E',
                 4.0: 'D',
                 4.5: 'C',
                 5.0: 'B',
                }

base_dict = {
              -2.0: 'D', 
              -1.5: 'C',
              -1.0: 'B',
              -0.5: 'A',
                 0: 'G',
               0.5: 'F',
               1.0: 'E',
               1.5: 'D',
               2.0: 'C',
               2.5: 'B',
               3.0: 'A',
               3.5: 'G',
               4.0: 'F',
               4.5: 'E',
               5.0: 'D',
            }


''' creating a class to store the templates as object '''
class note:

    def __init__(self, x, y, width, height, symbol_type, confidence = 0, pitch = '_') :
        self.x = x  #start position 
        self.y = y  #start position
        self.width = width
        self.height = height
        self.symbol_type = symbol_type
        self.confidence = confidence
        self.pitch = pitch

    def print(self):
        return f"{self.x} {self.y} {self.height} {self.width} {self.symbol_type} {self.pitch} {self.confidence}"


''' preprocessing'''
def preprocess_image(img):
    # convert image to grayscale
    im = img.convert('L')
    im_ = np.array(im)
    return 1 - im_ / 255.0


''' perform cross_correlation '''
@jit
def cross_correlation(music_, template_):
    # for keeping the cross_correlation operation values
    scores = np.zeros((music_.shape[0], music_.shape[1]))
    print('performing cross_correlation... may take a while')
    start = time.time()

    for y in range(music_.shape[0] - template_.shape[0]):
        for x in range(music_.shape[1] - template_.shape[1]):
            score = 0
            for j in range(template_.shape[0]):
                for i in range(template_.shape[1]):
                    score +=  music_[y+j][x+i] * template_[j][i]
            scores[y][x] = score
    
    end = time.time()
    print(round((end - start)/60.0, 2))
    return scores


''' get notes based on confidence '''
# @jit
def get_notes(music_, template_, scores, symbol_type, confidence_level = .9):
    print('get notes')
    notes = []
    max_score = scores.max()
    height = template_.shape[0]
    width = template_.shape[1]

    for y in range(music_.shape[0]):
        for x in range(music_.shape[1]):
            if scores[y][x] > confidence_level * max_score:
                notes.append(note(x, y, width, height, symbol_type, round((scores[y][x] / max_score * 100.0), 2)))

    return notes


''' Calculate the overlap between two notes using the intersection over union '''
def calculate_overlap(note1, note2):

    x1 = max(note1.x, note2.x)
    y1 = max(note1.y, note2.y)
    x2 = min(note1.x + note1.width, note2.x + note2.width)
    y2 = min(note1.y + note1.height, note2.y + note2.height)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = note1.width * note1.height + note2.width * note2.height - intersection

    return intersection / union


''' remove redundant bounding boxes '''
# @jit
def non_max_suppression(notes, threshold = .05):
    print('perform non_max_suppression')
    # sort notes by decreasing height
    notes = sorted(notes, key = lambda x : x.height, reverse= True)

    # create a list to store the selected notes
    selected_notes = []

    # Loop through each note in the sorted list
    for i, note in enumerate(notes):
        # Add the first note to the selected list
        if i == 0:
            selected_notes.append(note)
        else:
            overlaps = [calculate_overlap(note, n) for n in selected_notes]
            # If the maximum overlap is below the threshold, add the current note to the selected list
            if max(overlaps) < threshold:
                selected_notes.append(note)

    return selected_notes


''' calculate the pitch of the filled_note for treble and base '''
# @jit
def populate_pitch(notes, dis, treble_pos_list, base_pos_list, space):

    for note in notes:
        note_pos = note.y
        vis = 0

        for treble_pos in treble_pos_list:
            if abs(note_pos - treble_pos) < 8 * space:
                vis = 1
                ratio = round((note_pos - treble_pos) / dis, 2)
                min_dis = 55
                key = ''
                for i in range(-20, 55, 5):
                    window = i / 10
                    if abs(ratio - window) < min_dis:
                        min_dis = abs(ratio - window)
                        key = i / 10
                if key in base_dict:
                    note.ratio = ratio
                    note.pitch = treble_dict[key]
                    break
        if vis == 0:
            for base_pos in base_pos_list:
                if abs(note_pos - base_pos) < 8 * space:
                    ratio = round((note_pos - base_pos) / dis, 2)
                    min_dis = 55
                    key = ''
                    for i in range(-20, 55, 5):
                        window = i / 10
                        if abs(ratio - window) < min_dis:
                            min_dis = abs(ratio - window)
                            key = i / 10
                    if key in base_dict:
                        note.pitch = base_dict[key]
                        break

    return notes


''' draw boundaries where the template is detected '''
def draw_rectangles(music, notes, color):
    
    font = ImageFont.truetype("Roboto-Black.ttf", size=20)
    draw = ImageDraw.Draw(music)
    for note in notes:
        draw.rectangle([note.x, note.y, note.x + note.width,  note.y + note.height], outline=color)
        if note.symbol_type == 'filled_note':
            draw.text((note.x - 15, note.y - 6), note.pitch, fill='red', font=font)
    return music


''' populate the details of each note in a text file'''
def write_to_txt(filename, notes):

    # Open the file in write mode
    file = open(filename, "w")

    for note in notes:
    # Write values to the file
        file.write(note.print())
        file.write("\n")

    # Close the file
    file.close()


if __name__ == "__main__":

    music_path = str(sys.argv[1])
    start = time.time()

    music = Image.open(music_path)
    if music.mode == 'L':
        music = music.convert('RGB')
    music_ = preprocess_image(music)
    notes1 = []
    notes2 = []
    notes3 = []
    
    # get the list of the initial postions of all the treble and base
    treble_pos_list, base_pos_list, space = staffDetection.get_treble_base(music)
    print(treble_pos_list)
    print(base_pos_list)
    print(space)

    # for template 1
    print('Processing for template 1')
    template = Image.open(template_path_1)
    template = template.resize((template.width, space), Image.ANTIALIAS)

    template_ = preprocess_image(template)
    scores = cross_correlation(music_, template_)
    notes = get_notes(music_, template_, scores, 'filled_note', .9)
    notes1 = non_max_suppression(notes)
    notes1 = populate_pitch(notes1, template_.shape[0], treble_pos_list, base_pos_list, space)
    music = draw_rectangles(music, notes1, "red")

    # for template 2
    print('Processing for template 2')
    template = Image.open(template_path_2)
    template_ = preprocess_image(template)
    
    scores = cross_correlation(music_, template_)
    notes = get_notes(music_, template_, scores, 'quarter_rest', .99)
    notes2 = non_max_suppression(notes)
    music = draw_rectangles(music, notes2, "green")

    # for template 3
    print('Processing for template 3')
    template = Image.open(template_path_3)
    template_ = preprocess_image(template)

    scores = cross_correlation(music_, template_)
    notes = get_notes(music_, template_, scores, 'eighth_rest', .85)
    notes3 = non_max_suppression(notes)
    music = draw_rectangles(music, notes3, "blue")

    music.save('detected.png')
    music.show('detected.png')
    write_to_txt('detected.txt', notes1 + notes2 + notes3)

    end = time.time()
    print(end - start)
