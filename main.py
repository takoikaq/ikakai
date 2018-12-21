#ディープラーニング関連
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras import optimizers
from keras.models import load_model
#import cv2
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image


#Flask関連
from flask import Flask, render_template, request, url_for
from werkzeug import secure_filename #ヴェルクツォイクと発音


#学習済みのディープラーニングモデルを再構築する
#vgg16を構築
#input_tensorの定義
input_tensor = Input(shape=(75, 75, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

#ローカルモデルを構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:])) 
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

#学習済みの重みをロード
top_model.load_weights('./my_model_weights.h5')

#コンパイルする
top_model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
            metrics=['accuracy'])


#VGG16とローカルモデルを接続
model = Model(input=vgg16.input, output=top_model(vgg16.output))



graph = tf.get_default_graph()
#flask
app = Flask(__name__)

#アップロードされたファイルを格納するためのフォルダを作る
UPLOAD_FOLDER = './uploads' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods = ['GET', 'POST']) #アクセス時(GET)とファイル送信時(POST)で処理を分ける
def upload_file(): #アクセス時に関数を実行
    
    if request.method == 'GET': #アクセス時
        
        return render_template('index.html') #index.htmlを表示
    
    
    elif request.method == 'POST': #ファイル送信時
        f = request.files['file'] #アップロードされた画像をファイルオブジェクトとして格納
        filepath = UPLOAD_FOLDER + secure_filename(f.filename) #ファイルパスを作る
        f.save(filepath) #ファイルパスの保存
        #img = cv2.imread(filepath) #画像の読み込み
        #b,g,r = cv2.split(img) #b,g,r順となっているのでr,g,bに変換する
        #img = cv2.merge([r,g,b]) #画像をr,g,b順にする
        #img = cv2.resize(img, (75,75))

        form_img = Image.open(f)
        img_rev = form_img.resize((75, 75))
        img_rev=img_rev.convert('RGB')
        print(img_rev)
        x = image.img_to_array(img_rev)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0
        
        global graph
        #スレッド間を共有するため、graphを開きます
        with graph.as_default():
            score = model.predict(x)
            print(score)
            pred = np.argmax(score)            
                        
            #score = model.predict(np.array([img]))
            #pred = np.argmax(score[0])
        
        #result.htmlに値を渡し判別結果を表示する。判別結果は%で表現したいのでscoreは100倍にしてから渡す。"""
        return render_template('result.html', pred = pred, score = score * 100 )



@app.route('/result')
def result():
    return render_template('result.html')
        
        

#アプリを実行
if __name__ == '__main__':
    #app.debug = True #本番ではデバッグモードをオフにする。
    #app.run(host='0.0.0.0') #どこからでもアクセス可能にする。
    app.run()
            
        
