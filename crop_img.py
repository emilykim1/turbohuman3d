from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os
import tqdm

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

def align_face_crop(real, scale=7.0, size=448, pad_color=(0,0,0)):
    faces = app.get(real)
    if len(faces) == 0:
        # print(f"No face found in {image_path}")
        return []

    # Use aligned face, then expand around it
    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    box_size = int(max(x2 - x1, y2 - y1) * scale)

    # Compute square crop centered on face
    crop_x1 = cx - box_size // 2
    crop_y1 = cy - box_size // 2 + box_size // 4
    crop_x2 = cx + box_size // 2
    crop_y2 = cy + box_size // 2 + box_size // 4
    
    output = []
    
    for img in [real]:

        crop = img[max(0, crop_y1):crop_y2, max(0, crop_x1):crop_x2]
        h, w = crop.shape[:2]
        
        desired = np.full((crop_y2 - crop_y1, crop_x2 - crop_x1, 3), pad_color, dtype=np.uint8)
        desired[max(0, -crop_y1):max(0, -crop_y1) + h, max(0, -crop_x1): max(0, -crop_x1) + w] = crop
        crop = desired

        # Pad to preserve aspect ratio
        h, w = crop.shape[:2]
        target_w, target_h = (size,size)
        scale_factor = min(target_w / w, target_h / h)
        resized_crop = cv2.resize(crop, (int(w * scale_factor), int(h * scale_factor)))

        # Center pad to target size
        padded = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        new_h, new_w = resized_crop.shape[:2]
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_crop
        output += [padded]

    # cv2.imwrite(output_path, crop)
    return output

if __name__=='__main__':
    base_dir = '/home/emilykim/Desktop/HAR-project/training_data/'
    seqs = os.listdir(base_dir)
    seqs = [seq for seq in seqs if 't1+' in seq]
    seqs.sort()
    out_dir = '/home/emilykim/Desktop/HAR-project/training_data-cropped-rmbg-1/'
    os.makedirs(out_dir, exist_ok=True)
    
        
    for seq in seqs:
        print(seq)
        reals = os.listdir(base_dir + f'{seq}/real/')
        # syns = os.listdir(base_dir + '/syn/')
        os.makedirs(f'{out_dir}/{seq}/syn/', exist_ok=True)
        os.makedirs(f'{out_dir}/{seq}/real/', exist_ok=True)
        # print(seq)
        for real in tqdm.tqdm(reals):
            real_img = cv2.imread(base_dir + '/' + seq + f'/real/{real}')
            syn_img = cv2.imread(base_dir + '/' + seq + f'/syn/{int(real[:-4]):05d}.png')
            # print(real_img, syn_img)
            output_real = align_face_crop(real_img)
            output_syn = align_face_crop(syn_img)
            if len(output_real) == 0 or len(output_syn) == 0:
                continue
            # real_img, syn_img = output[0], output[1]
            cv2.imwrite(out_dir + '/' + seq + f'/real/{real}', output_real[0])
            cv2.imwrite(out_dir + '/' + seq + f'/syn/{int(real[:-4]):05d}.png', output_syn[0])
            