from flask import Flask, render_template, Response

import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


img1 = cv2.imread("swagg2.png")
import copy
def getPerpCoord(a, b, length):
    [aX, aY] = a
    [bX, bY] = b
    vX = bX-aX
    vY = bY-aY
    #print(str(vX)+" "+str(vY))
    if(vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return [int(cX), int(cY), int(dX), int(dY)]
def triangles(points):
    points = np.where(points, points, 1)
    print(points)
    subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
    for p in list(points):
        pt = tuple([int(round(p[0]) ), int(round( p[1] )) ])
        subdiv.insert(tuple(pt))
    for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
        yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]

def crop(img, pts):
    x, y, w, h = cv2.boundingRect(pts)
    img_cropped = img[y: y + h, x: x + w]
    pts[:, 0] -= x
    pts[:, 1] -= y
    return img_cropped, pts

def warp(img1, img2, pts1, pts2): 
    for indices in triangles(pts1):
        img1_cropped, triangle1 = crop(img1, pts1[indices])
        img2_cropped, triangle2 = crop(img2, pts2[indices])
        transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
        img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        mask = np.zeros_like(img2_cropped)
        cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
        img2_cropped *= 1 - mask
        img2_cropped += img2_warped * mask

app = Flask(__name__)
cap = cv2.VideoCapture(0) 

def gen_frames():
    with mp_pose.Pose(
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue


            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img = image
            xpos = []
            ypos = []    
            p_landmarks = results.pose_landmarks
            if p_landmarks:
                vis = []
                for i in str(p_landmarks).split('landmark')[1:26]:
                    i_1 = i.split()
                    xpos.append(int(640 *  float(i_1[2])))
                    ypos.append(int(480 *  float(i_1[4])))
                    vis.append(float(i_1[8]))
                print(min(vis))
                minx = min(xpos)
                maxx = max(xpos)
                miny = min(ypos)
                maxy = max(ypos)
                image = img
            array = np.zeros([480, 640, 3],
                        dtype = np.uint8)
            array[:, :] = [255, 255, 255]
            if len(xpos) > 24:
                print('a')
                r_elbowcoords = [xpos[14], ypos[14]]
                r_shouldercoords = [xpos[12], ypos[12]]
                l_elbowcoords = [xpos[13], ypos[13]]
                l_shouldercoords = [xpos[11], ypos[11]]
                l1 = getPerpCoord(r_shouldercoords, r_elbowcoords, 5)
                l2 = getPerpCoord(l_shouldercoords, l_elbowcoords, 5)
                ipos1 = [l1[0], l1[1]]
                
                ipos3 = [xpos[12], ypos[12] - 20]
                ipos4 = [(xpos[11] + xpos[12]) / 2, ((ypos[11] + ypos[12]) / 2) - 40]
                ipos5 = [xpos[11], ypos[11] - 20]
                
                ipos7 = [l2[2], l2[3]]
                ipos8 = [xpos[24] - 40, ypos[24]]
                ipos9 = [xpos[23] + 40, ypos[23]]        
                pts1 = np.array([[70,294], [254,64], [421,22], [587,64], [762,293], [256,548], [567,548]])
                pts2 = np.array([[int(i[0]), int(i[1])] for i in [ipos1,ipos3,ipos4,ipos5,ipos7,ipos8,ipos9]])
                try:
                    warp(img1, array, pts1, pts2)
                except:
                    print('deez')

            h, w, c = array.shape   
            image_bgra = np.concatenate([array, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
            white = np.all(array == [255, 255, 255], axis=-1)   
            image_bgra[white, -1] = 0
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            alpha_background = image[:,:,3] / 255.0
            alpha_foreground = image_bgra[:,:,3] / 255.0

            # set adjusted colors
            for color in range(0, 3):
                image[:,:,color] = alpha_foreground * array[:,:,color] + \
                    alpha_background * image[:,:,color] * (1 - alpha_foreground)

            # set adjusted alpha and denormalize back to 0-255
            image[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 5000, debug=True)