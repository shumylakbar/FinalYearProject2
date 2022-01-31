import sys
import time
from multiprocessing import Process
import pickle
import speech_recognition as sr  # used for speech to text
import pyttsx3  # used for text to speech
from typing import TextIO
import nltk  # Natural Language Toolkit
import speech_recognition as sr  # used for speech to text
import numpy    # list and array handling
import glob     # the glob module is used to retrieve files/path names matching a specified pattern
import tflearn  # transparent deep learning library built on top of Tensorflow.
import json     # JavaScript Object Notation
import pickle   # pickle module is used for serializing and de-serializing a Python object structure
import random   # chose or generate random choices
import pyttsx3  # used for text to speech
import wikipedia  # wikipedia handling
import datetime   # time and date module
import tensorflow
from selenium import webdriver  # Selenium uses webserver for handling data from website
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from pygame import mixer  # game or music playing
from tensorflow.python.framework import ops  # a Python library for fast numerical computing
from nltk.stem.lancaster import LancasterStemmer  # NLP usage for dropping the suffices
import cv2
import face_recognition
import numpy as np
import csv
from datetime import date
from datetime import datetime
from fuzzywuzzy import fuzz


PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)
act = ActionChains(driver)

stemmer = LancasterStemmer()
engine = pyttsx3.init()
r = sr.Recognizer()

allName = []

words = []
labels = []
docs_x = []
docs_y = []
check = []
check2 = []
voice_x = int(0)
confusion = ['Sir would you please repeat', 'I did not understand',
             'Speak English please', 'Say that in english please',
             'I understand English language only']
with open("intents.json") as file:
    data = json.load(file)


def check_for_updates():
    for i in data["intents"]:
        for p in i["patterns"]:
            check.append(p)
    try:
        with open("ppp.pickle", "rb") as pop:
            check2.extend(pickle.load(pop))
        if check == check2:
            print('all okay')
        else:
            print('!!! Data has been modified Please wait for a sec... ')
            #print(error)
    except:
        with open("ppp.pickle", "wb") as pop:
            pickle.dump((check), pop)


check_for_updates()

try:
    if check == check2:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    else:
        raise ValueError

except:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            docs_x.append(word)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    words = [stemmer.stem(w.lower()) for w in words if w not in '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []

        word = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in word:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)


# --------------------------------------------------------------
#print(output)
# tensorflow.reset_defualt_graph()
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    if check == check2:
        model.load("model.tflearn")
    else:
        raise ValueError
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


with open('names_0.csv') as allMuslimNames:
    reader = csv.reader(allMuslimNames)
    for row in reader:
        allName.append(row[0].lower())
    allMuslimNames.close()

with open("chkTrFl.pickle", "wb") as pop:
    pickle.dump(('', '', []), pop)
    pop.close()


def date_time_2day():
    today_date = date.today()
    d1 = today_date.strftime("%d/%m/%Y")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return str(d1) + ' ' + '(' + str(current_time) + ')'


def name_match_improve(c):
    li = []
    for i in c:
        match = []
        for x in allName:
            num3 = fuzz.ratio(i, x)
            match.append(num3)

        short_d_index = sorted(range(len(match)), key=lambda k: match[k])
        string = allName[short_d_index[-1]]
        print(string)
        li.append(string)

    li = ' '.join(li)
    return li


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


def text_to_speech(receive_text):  # Speech to text function
    engine.say(receive_text)
    engine.runAndWait()
    return receive_text


def speech_to_text():
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            text_to_speech('OK')
            # r.pause_threshold = 1  # seconds of non speaking audio before phrase is considered complete.
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)

            try:
                print("Recognizing...")
                #  query = r.recognize_sphinx(audio, language='en-IN')  # Using google for voice recognition.
                # query = r.recognize_sphinx(audio)  # Using google for voice recognition.
                query = r.recognize_google(audio, language='en-in')  # Using google for voice recognition.
                print(f"User said: {query}\n")  # User query will be printed.
                return query

            except:
                return ''


def virtual_assistant(query):
    if 'wikipedia' in query:
        text_to_speech('What do you want me to search in wikipedia Sir tell me only the topic name for example "India"?')
        query = speech_to_text()
        text_to_speech('Searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=1)
        text_to_speech("According to Wikipedia")
        print(text_to_speech(results))
        # return 'are you listening ?'

    elif 'facebook' in query:
        driver.get("https://www.facebook.com")
        print(driver.title)  # websitename
        driver.find_element_by_name("email").send_keys("Email@yahoo.com")
        driver.find_element_by_name("pass").send_keys("12345")
        act.send_keys(Keys.RETURN).perform()
        # return 'tell me more'

    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        print(text_to_speech(strTime))
        # return 'are you listening ?'

    elif 'date' in query:
        strTime = datetime.datetime.now().strftime("%A:%d:%B:%Y")
        print(text_to_speech(strTime))
        # return 'are you listening ?'
    else:
        return


def entertainment(query):
    if 'song' in query:
        text_to_speech("Which song do you want to play")
        files_list = glob.glob('E:\My Codes of python\song/*.mp3')
        for item in files_list:
            print(item)
        print(files_list)
        while True:
            query = speech_to_text().lower()
            for song in files_list:
                if query in song:
                    print("playing", song)
                    mixer.init()
                    mixer.music.load(song)
                    mixer.music.play()
                elif 'stop' in query or 'pause' in query:
                    mixer.music.pause()
                elif "start" in query or 'play' in query:
                    mixer.music.unpause()
                elif "break" in query or 'exit' in query:
                    mixer.music.stop()
                    break

        # return 'are you there'
    else:
        text_to_speech('Did you enjoy the song')
        return


def function1():
    cap = cv2.VideoCapture(0)
    checktig = 0
    todaysVisitedClient = []


    while True:
        with open("faceRecord.pickle", "rb") as f:
            encodeListKnown, imageName = pickle.load(f)
        success, frame = cap.read()  # imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            y1, x2, y2, x1 = faceLoc
            varBox = y2 + int((y2 - y1) * 0.16)
            varTxt = y2 + int((y2 - y1) * 0.134)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = imageName[matchIndex]
                if name.lower() not in todaysVisitedClient:
                    todaysVisitedClient.append(name.lower())
                    modification_data_appand_dateTime(name.lower())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, varBox), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 3, varTxt), cv2.FONT_HERSHEY_SIMPLEX, (y2 - y1) * 0.0054, (0, 0, 0))
                with open("chkTrFl.pickle", "wb") as p_known:
                    pickle.dump(('True', name, []), p_known)
                    p_known.close()
                if checktig == 6:
                    checktig = 0

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, varBox), (x2, y2), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, 'unknown', (x1 + 3, varTxt), cv2.FONT_HERSHEY_SIMPLEX, (y2 - y1) * 0.0054, (0, 0, 255))
                if checktig <= 5:
                    print(checktig, ' ...')
                    if checktig == 5:
                        with open("chkTrFl.pickle", "wb") as p_unknown:
                            pickle.dump(('False', 'unknown', encodeFace), p_unknown)
                            p_unknown.close()
                    checktig += 1

        cv2.imshow('webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            with open('namesClient_visited_today.csv', 'a+', newline='') as writeObject:
                csv_writer = csv.writer(writeObject)
                todaysVisitedClient.append(date_time_2day())
                csv_writer.writerow(todaysVisitedClient)
                writeObject.close()
            break
    cap.release()
    cv2.destroyWindow("cap")


def modification_data_appand_dateTime(name):
    g = []
    print('i found...........')
    with open('client_records.csv') as csvFileTemp:
        reader = csv.reader(csvFileTemp)
        for row in reader:
            g.append(row)

    for x in range(0, len(g)):
        if name == g[x][0]:
            g[x].append(date_time_2day())
            print(g[x])

    with open('client_records.csv', 'w', newline='') as fileTemp:
        csv_writer = csv.writer(fileTemp)
        csv_writer.writerows(g)
        fileTemp.close()

    with open('new_names.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(g)
        file.close()


def modification_data_new_entry(name, encodings):
    tempList = []
    tempName = []
    data_csv = []

    with open("chkTrFl.pickle", "wb") as pop:
        pickle.dump(('', '', []), pop)
        pop.close()

    with open("faceRecord.pickle", "rb") as f:
        encodeListKnown, imageName = pickle.load(f)
        f.close()

    for x in imageName:
        tempName.append(x)
    for x in encodeListKnown:
        tempList.append(x)
    tempName.append(name)
    tempList.append(encodings)
    # saving the new client name and face math in faceRecord.pickle file
    with open("faceRecord.pickle", "wb") as f:
        pickle.dump((tempList, tempName), f)
        f.close()

    # saving the name in client record file as well
    with open('client_records.csv', 'a+', newline='') as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerow([name])
        write_obj.close()
    with open('client_records.csv') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data_csv.append(row)
    with open('new_names.csv', 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(data_csv)
        file.close()


def function2():
    sys.stdin = open(0)
    print(text_to_speech("Chat Bot Activated !"))
    y = x = 0
    while True:
        time.sleep(0.1)
        with open("chkTrFl.pickle", "rb") as f:
            checktig, checkname, encodings = pickle.load(f)
            f.close()

        if checktig == 'False':
            try:
                print(text_to_speech('your face is detected unknown'))
                print(text_to_speech('Please tell me your name:'))
                query = speech_to_text().lower()
                if not query:
                    raise ValueError('')
                elif 'break' in query:
                    break
            except ValueError as e:
                print(text_to_speech('Speak please'))
                continue

            else:
                c = list(query.split(" "))
                qName = name_match_improve(c)
                while True:
                    if x == 0:
                        print(text_to_speech('Did You say ' + qName))
                    else:
                        print(text_to_speech('please spell out your name: '))

                    query = speech_to_text().lower()

                    if (x == 1) and (query != ''):
                        a = query.replace(" ", "")
                        print(text_to_speech('Did You say ' + a))
                        query = speech_to_text().lower()
                        if 'yes' in query or 'right' in query or 'correct' in query or 'that' in query or 'my name' in query:
                            print(text_to_speech('Okay ' + a + ' your name has been saved'))
                            modification_data_new_entry(a, encodings)
                            x = 0
                            break
                        else:
                            break
                    try:
                        if 'yes' in query or 'right' in query or 'correct' in query or 'that' in query or 'my name' in query:
                            print(text_to_speech('Okay ' + qName + ' your name has been saved'))
                            modification_data_new_entry(qName, encodings)
                            break
                        elif not query:
                            print(text_to_speech('talk please'))
                            raise ValueError(' ')
                        elif 'no' in query or 'spelling' in query:
                            x = 1
                            continue
                    except ValueError as e:
                        continue

        elif checktig == 'True':
            # print(text_to_speech('hello ' + checkname))
            try:
                query = speech_to_text().lower()
                if not query:
                    print(text_to_speech('i can see you, you can talk to me'))
                    raise ValueError(' ')
                elif 'stop' in query or 'goodbye' in query or 'good bye' in query:
                    text_to_speech("See you Later " + checkname)
                    break
                # elif 'wikipedia' in query or 'facebook' in query or 'time' in query or 'date' in query:
                #     query = virtual_assistant(query)
                # elif 'song' in query:
                #     query = entertainment(query)
                elif 'change your voice' in query or 'change voice' in query or 'female voice' in query:
                    y = (y + 1) % 2
                    # print('my voice = ', y)
                    voices = engine.getProperty('voices')
                    engine.setProperty('voice', voices[y].id)
                    text_to_speech('my voice has been changed now ' + checkname)
                else:
                    results = model.predict(([bag_of_words(query, words)]))[0]
                    results_index = numpy.argmax(results)
                    tag = labels[results_index]

                    if results[results_index] > 0.8:
                        for tg in data["intents"]:
                            if tg['tag'] == tag:
                                responses = tg['responses']
                        print(text_to_speech(random.choice(responses)))
                    else:
                        print(text_to_speech(random.choice(confusion)))
                        # print(text_to_speech('Did you say something ?'))

            except ValueError as e:
                continue

        else:
            pass
        # time.sleep(1)


if __name__ == "__main__":
    p1 = Process(target=function1)
    p2 = Process(target=function2)
    p1.start()
    p2.start()



























