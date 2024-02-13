# KantanKaiwa_Style-Bert-VITS2
Style-Bert-VITS2モデルと簡単な音声会話をするためのスクリプトです。（適宜改造してください）  
ただ設定通りの振る舞いをしてもらいつつ音声会話をするためだけのスクリプトになっています。  

## 使用例
動画ではgpt-3.5-turbo-0125を使用していました。  
https://www.youtube.com/embed/fkyQgcnX68U?si=rFXhPkBk6emLyCGU  

## 使い方
Style-Bert-VITS2のserver_fastapi.pyを実行してから、このスクリプトをターミナル上で実行してください。  
設定（params）のところが自分用のままなので、適宜値を変更してみてください。　　
（特にmodel_idは動かしたいモデルに変更してください。）  
ターミナル上で"go"と打つと録音が開始し、設定秒数無音を検出すると録音が終了し、whisperで文字起こしがされます。whisperで文字起こしされた情報がuser_input変数に代入され、OpenAIのChatモデルに送信、返ってきた文章をStyle-Bert-VITS2でTTS変換します。

## 便利リポジトリの紹介
https://github.com/p2-3r/Discord-ReadTextBot-for-Style-Bert-VITS2-API  
discord bot上でStyle-Bert-VITS2を動かすためのものなのですが、パスを設定しておけばrun_API.batで仮想環境ごとserver_fastapi.pyを起動できます。

