import os
import cv2
from PIL import Image, ImageOps

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

def check_folders():
    paths = {
        'data_path' : 'data',
        'images_path' : 'data/images',
        'videos_path' : 'data/videos'
        
    }
    # Check whether the specified path exists or not
    notExist = list(({file_type: path for (file_type, path) in paths.items() if not os.path.exists(path)}).values())
    
    if notExist:
        print(f'Folder {notExist} does not exist. We will created')
        # Create a new directory because it does not exist
        for folder in notExist:
            os.makedirs(folder)
            print(f"The new directory {folder} is created!")
  
def check_labels(model, image):
    result = model.predict(source = image)

    for result in result:
        boxes = result.boxes.cpu().numpy()
        # print(boxes)
        
        output_class = int(boxes.cls[0])
        print(output_class)
        class_name = model.names[output_class]
        print(class_name)
        bx1 = boxes.xywh.tolist()
        bx2 = boxes.xyxy.tolist()

        print(bx1[0])
        print(bx2[0])
        return image, bx1, bx2
        
def image_emotion(emotion, emotion_img, bx1, bx2):
    emotion_class = {'anger':[300,100,70,-450], 'anxiety':[600,600,-300,-300], 'embarrass':[-200,-600, 800,-100],'happy':[600,600,-300,-300],'normal':[],'pain':[100,-200,-50,0],'sad':[600,600,-300,-300]}
    img1 = Image.open(emotion_img)   
    img1 = ImageOps.exif_transpose(img1)
    print(img1.size)
    print(f'wh:{int(bx1[0][2])}, {int(bx1[0][3])}')
    print(f'xy:{int(bx2[0][0])}, {int(bx2[0][0])}')
    image_file = {'anger':'anger_small.png', 'anxiety':'axiety.png', 'embarrass':'sweat_line.png','happy':'happy_frame_3.png','normal':'','pain':'bandage_mung.png','sad':'sad2.png'}

    sweat_frame = f'./data/decoration/{emotion}/{image_file[emotion]}'
    img2 = Image.open(sweat_frame) 
    new_size = (int(bx1[0][2])+emotion_class[emotion][0], int(bx1[0][3])+emotion_class[emotion][1])
    img2_resized = img2.resize(new_size)
    img1.paste(img2_resized, (int(bx2[0][0])+emotion_class[emotion][2], int(bx2[0][1])+emotion_class[emotion][3]), mask=img2_resized)

    return img1