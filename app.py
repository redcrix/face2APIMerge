from fastai.vision import *
from flask import Flask, json, request
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
import re
import time
from passporteye import read_mrz
import datetime
from dateutil import relativedelta
from datetime import datetime as dt
import face_recognition

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

path = 'model_id/'
learn = load_learner(path, 'id_card.pkl')
learn = learn.load('pystage-2')

app = Flask(__name__)


@app.route("/front", methods=['POST', 'GET'])

def cards():
    time_str = time.strftime("%Y%m%d-%H%M%S")
    if request.method == 'POST':
        print(request.files)
        if 'idfront' not in request.files:
            return json.dumps({"message": 'No image of card', "success": False, "code": 201})
        idfront = request.files['idfront']
        idfront.save('images/'+time_str + '_front_card10.jpg')
        img = open_image('images/'+time_str + '_front_card10.jpg')

        if 'fullname' not in request.form:
            return json.dumps({"message": 'No Name Found', "success": False, "code": 201})
        fullname = request.form['fullname']

        if 'p_image' not in request.files:
            return json.dumps({"message": 'No profile image of user', "success": False, "code": 201})
        p_image = request.files['p_image']
        p_image.save('images/'+time_str + '_p_image10.jpg')

        pred_class, pred_idx, outputs = learn.predict(img)
        array = pred_idx.tolist()

    if array == 0:    #driving card

        alpha = 1
        beta = 25

        img1 = cv2.imread('images/'+time_str + '_front_card10.jpg')
        result = cv2.addWeighted(img1, alpha, np.zeros(img1.shape, img1.dtype), 0, beta)
        text = (pytesseract.image_to_string(img1))
        number = re.findall(r"[0-9]{12}", text)
        dob = re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)
        wa2 = datetime.datetime.strptime(dob, "%d/%m/%Y").strftime("%Y,%m,%d")
        f1 = int(wa2[:4])
        a1 = dt(f1, 1, 1)
        b1 = dt.today()
        delta = relativedelta.relativedelta(b1, a1)
        years = delta.years

        name = re.findall(r"[A-Z]{2,15}\s[A-Z]{2,15}\s[A-Z]{2,15}", text)
        if fullname == name[1]:
            c1 = True
        else:
            c1 = False

        img2 = cv2.imread('images/'+time_str + '_front_card10.jpg')
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 7)

        for (x, y, w, h) in faces:
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = img2[y:y + h, x:x + w]
            crop_pic = cv2.imwrite('images/'+time_str +'_croppic10.jpg', roi_color)

        known_image = face_recognition.load_image_file('images/'+time_str + '_p_image10.jpg')
        unknown_image = face_recognition.load_image_file('images/'+time_str + '_croppic10.jpg')
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        bide_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)

        if years < 18:
            return json.dumps({"message": 'User is less than 18 years old', "success": False, "code": 201})

        if (results23 == {True}) and (fullname == name[1]):
            data = dict(
                        {"name": name[1], "dob": dob[0], "idn": number, "exp_date": dob[1], "valid_date": dob[2]})
            return json.dumps({"code": 200, "message": 'ID data extract successfully', "success": True,
                                       "data": data}).replace("[", "").replace("]", "").replace(" PRIA", "").replace(" PRA", "")
        elif results23 == [True] and c1 == False:
            return json.dumps({"message": 'Full name does not match ', "success": False, "code": 201})
        elif results23 == [False] and c1 == True:
            return json.dumps({"message": 'Profile Pic does not match ', "success": False, "code": 201})

        else:
            data = dict({"name": '', "dob": '', "idn": '', "match": False})
            return json.dumps({"code": 201, "message": 'ID data extract unsuccessfully', "success": False, "data": data})

    elif array == 1:     #National Card

        img3 = cv2.imread('images/'+time_str + '_front_card10.jpg')
        gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 7)

        for (x, y, w, h) in faces:
            ix = 0
            cv2.rectangle(img3, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = img3[y:y + h, x:x + w]

            crop_pic = cv2.imwrite('images/'+time_str + '_croppic10.jpg', roi_color)

        known_image = face_recognition.load_image_file('images/'+time_str + '_p_image10.jpg')
        unknown_image = face_recognition.load_image_file('images/'+time_str + '_croppic10.jpg')
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        bide_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)

        alpha = 1
        beta = 25
        result = cv2.addWeighted(img3, alpha, np.zeros(img3.shape, img3.dtype), 0, beta)

        text = pytesseract.image_to_string(img3)

        text = text.replace("/\s+/g, ' '", "")
        # print(text)
        number = re.findall(r"[\+\(]?[1-9][0-9 .\-()]{8,}[0-9]", text)
        date = str(re.findall(r"[\d]{1,4}[/-][\d]{1,4}[/-][\d]{1,4}", text)).replace("[", "").replace("]","").replace("'", "")
        print(date)

        wa12 = datetime.datetime.strptime(date, "%d/%m/%Y").strftime("%Y,%m,%d")
        f2 = int(wa12[:4])
        a1 = dt(f2, 1, 1)
        b1 = dt.today()
        delta = relativedelta.relativedelta(b1, a1)
        years = delta.years

        name = re.findall(r"[A-Za-z]{2,15}\s[A-Za-z]{2,15}", text)  # ^[
        print(name)

        if fullname == name[0]:
            c1 = True
        else:
            c1 = False

        if years < 18:
            return json.dumps({"message": 'User is less than 18 years old', "success": False, "code": 201})

        if results23 == [True] and c1 == True:
            data = dict({"name": name[0], "dob": date, "idn": number, "match": True})
            return json.dumps({"code": 200, "message": 'ID data extract successfully', "success": True,
                                           "data": data}).replace("[", "").replace("]", "").replace("svat fa",
                                                                                                    "").replace(
                            "SRR swat", "").replace("afte ava", "")
        elif results23 == [True] and c1 == False:
            return json.dumps({"message": 'Full name does not match ', "success": False, "code": 201})
        elif results23 == [False] and c1 == True:
            return json.dumps({"message": 'Profile Pic does not match ', "success": False, "code": 201})
        else:
            data = dict({"name": '', "dob": '', "idn": '', "match": False})
            return json.dumps({"code": 201, "message": 'ID data extract unsuccessfully', "success": False,
                                           "data": data}).replace("[", "").replace("]", "")

    elif array == 2:    #passport card
        a = fullname
        mrz = read_mrz('images/'+time_str + '_front_card10.jpg')
        mrz_data = mrz.to_dict()

        n = mrz_data['names'].replace(" KK",'').replace("KK", '').replace(" K", '').replace(" ", '')
        n = n.capitalize()

        s = mrz_data['surname'].replace(" ", '')
        s = s.capitalize()
        o = n + " " + s
        #print(o)

        no = mrz_data['number'].replace("<", ' ').replace(" ", '')
        se = mrz_data['sex']
        c = mrz_data['country']

        dob = mrz_data['date_of_birth']
        dob = datetime.datetime.strptime(dob, "%y%m%d").strftime("%d/%m/%Y")
        wa2 = datetime.datetime.strptime(dob, "%d/%m/%Y").strftime("%Y,%m,%d")
        f1 = int(wa2[:4])
        a1 = dt(f1, 1, 1)
        b1 = dt.today()
        delta = relativedelta.relativedelta(b1, a1)
        years = delta.years

        exp = mrz_data['expiration_date']
        exp = datetime.datetime.strptime(exp, "%y%m%d").strftime("%d/%m/%Y")

        up2 = cv2.imread('images/'+time_str + '_front_card10.jpg')
        gray = cv2.cvtColor(up2, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.2, 7)

        for (x, y, w, h) in faces:
            cv2.rectangle(up2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = up2[y:y + h, x:x + w]
            crop_pic = cv2.imwrite('images/'+time_str +'_croppic10.jpg', roi_color)

        tup2 = cv2.imread('images/'+time_str + '_p_image10.jpg')

        known_image = face_recognition.load_image_file('images/'+time_str + '_p_image10.jpg')
        unknown_image = face_recognition.load_image_file('images/'+time_str + '_croppic10.jpg')
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
        bide_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results23 = face_recognition.compare_faces([bide_encoding], unknown_encoding)

        if years < 18:
            return json.dumps({"message": 'User is less than 18 years old', "success": False, "code": 201})

        if a == o:
            c = True
        else:
            c = False

        if results23 == [True] and c == True:
            data = dict(
                         {"name": n, "surname": s, "dob": dob, "idn": no, "sex": se, "country": c, "expiry_date": exp,
                          "match": True})
            return json.dumps(
                         {"code": 200, "message": 'Passport_ID data extract successfully', "success": True,
                          "data": data}).replace("\\", "").replace("<", "").replace(" <<", "")
        elif results23 == [True] and c == False:
            return json.dumps({"message": 'Full name does not match ', "success": False, "code": 201})
        elif results23 == [False] and c == True:
            return json.dumps({"message": 'Profile Pic does not match ', "success": False, "code": 201})
        else:
            data = dict(
                         {"name": '', "dob": '', "idn": '', "surname": '', "sex": '', "country": '', "expiry_date": '',
                          "match": False})
            return json.dumps(
                         {"code": 201, "message": 'Passport_ID data extract unsuccessfully', "success": False,
                          "data": data}).replace("[", "").replace("]", "").replace("\\", "")


@app.route("/")
def hello():
    return "FRONT CARD OCR"


if __name__ == "__main__":
    app.run(debug=True)
