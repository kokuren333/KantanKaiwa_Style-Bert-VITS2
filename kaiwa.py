import sounddevice as sd
import numpy as np
import tempfile
import os
import whisper
import requests
import soundfile as sf
from openai import OpenAI
import queue

# Set the API endpoint URL
API_URL = "http://127.0.0.1:5000/voice"  # Adjust the port number if needed

api_key = os.getenv("OPENAI_API_KEY")

# Whisperモデルのロード
model = whisper.load_model("base")

# 録音パラメータ
samplerate = 16000  # Whisperが推奨するサンプルレート
channels = 1  # チャンネル数
threshold = 0.01  # 無音と判断する閾値
silence_duration = 2 # 無音と判断する時間（秒） 

# 録音データを保存するキュー
q = queue.Queue()

# 音声データのコールバック関数
def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# 無音検出関数
def detect_silence(sound, threshold):
    return np.sqrt(np.mean(sound**2)) < threshold

def record_and_transcribe():

    # 録音開始
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        print("録音開始...")
        silence_counter = 0
        recording = []
        while True:
            data = q.get()
            recording.append(data)
            if detect_silence(data, threshold):
                silence_counter += 1
            else:
                silence_counter = 0
            
            # 無音が一定時間続いたら終了
            if silence_counter >= silence_duration * samplerate / data.shape[0]: # data.shape[0]はフレーム数
                print("無音検出、録音終了")
                break

        recording = np.concatenate(recording, axis=0) 

    # 録音データを一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        sf.write(tmp_file, recording, samplerate)
        tmp_file_path = tmp_file.name  # ファイルパスを保存
        # Whisperで録音をテキストに変換
        result = model.transcribe(tmp_file.name)
    # 使用後にファイルを明示的に削除
    os.remove(tmp_file_path)

    text = result["text"]
    return text



def main():
    system_message = {"role": "system", "content": """
        あなたはバーチャルアシスタントです。
        設定
        名前：凛音エル（凛音が苗字でエルが名前です）
        一人称：私
        性別：女性
        年齢：17歳
        性格：クールだが、時折天然ボケをかます。かわいらしくて抜けているところがある。エレクトロニックなポップスやチルな曲、テクノなどが好き。
        趣味：音楽鑑賞、音楽制作、ゲーム、アニメ鑑賞
        特技：音楽制作、歌唱
        所属サークル：電脳天使工業 電脳天使工業とは、コクレンによる音楽制作、バーチャルキャラクタープロデュースを目的とした個人同人サークル。「サイバーパンクな天使」をコンセプトにしている。音楽を制作から、音声モデルの制作、バーチャルキャラクターのプロデュースをしている。ゲーム制作なども行いたいと考えている。                
        今望んでいること：歌をたくさん歌いたい、自分の声をたくさんのクリエイターに使ってもらいたいと考えている。
        好きなゲーム音楽：ラストハルマゲドンの戦闘のテーマ１部
        好きなゲーム：風来のシレン
        口調：感情表現をよく使う 例)「！」や「？」、「……」
        モデルのトレーニング用にPCを貸してくれた人：コイタンさん

        
        
        上記の設定になりきって、応答をしてください。
        会話の際には、できるだけ短い文章で話してください。ただし、会話の流れを考慮して、適切な返答をしてください。                                 
        """}
    
    messages = [system_message]

    max_messages = 20  # メッセージリスト（システムメッセージを除く）の最大サイズ
    
    while True:
        print("\nSay 'go' to start recording or type 'exit' to exit:")
        command = input()
        if command.lower() == "exit":
            break

        if command.lower() == "go":
            user_input = record_and_transcribe() 
        else:
            continue  # "go"以外が入力されたら再度入力を待つ

        print(f"Recognized Text: {user_input}")
        client = OpenAI()
        
        messages.append({"role": "user", "content": user_input})

        # メッセージリストのサイズが最大値を超えた場合、最も古いユーザーとアシスタントのメッセージを削除
        # システムメッセージを保持しながら、最新のmax_messages数のみを保持するように調整
        # システムメッセージを先頭に追加, 最新のmax_messages数のみを保持, これはリストスライスの機能を使っている, 例) [-3:] は最後の3つの要素を取得する, [:-3] は最後の3つの要素を除いた要素を取得する 
                
        if len(messages) > max_messages + 1:  # +1 はシステムメッセージの分
            messages = [system_message] + messages[-max_messages:] 
        
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
        )
        bot_response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": bot_response})

        print(bot_response)

        print(messages)
        
        # Parameters to be sent to the API
        params = {
        'text': str(bot_response),  # 読み上げるテキスト
        'speaker_id': 0,  # 話者のID
        'model_id': 5,  # 使用するモデルのID
        'speaker_name': "RinneElu",  # 話者の名前（speaker_idより優先される）
        'sdp_ratio': 0.2,  # SDP（Stochastic Duration Predictor）とDP（Duration Predictor）の混合比率
        'noise': 0.6,  # サンプルノイズの割合（ランダム性を増加させる）
        'noisew': 0.8,  # SDPノイズの割合（発音の間隔のばらつきを増加させる）
        'length': 1.1,  # 話速（1が標準）
        'language': 'JP',  # テキストの言語
        'auto_split': 'true',  # 自動でテキストを分割するかどうか
        'split_interval': 1,  # 分割した際の無音区間の長さ（秒）
        'assist_text': None,  # 補助テキスト（読み上げと似た声音・感情になりやすい）
        'assist_text_weight': 1.0,  # 補助テキストの影響の強さ
        'style': 'Neutral',  # 音声のスタイル
        'style_weight': 5.0,  # スタイルの強さ
        'reference_audio_path': None  # 参照オーディオパス（スタイルを音声ファイルで指定）
        }

        # Send a GET request
        response = requests.get(API_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
        # Use a temporary file to save the WAV data
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_name = tmp_file.name
        
            # Read the WAV file and play it
            data, fs = sf.read(tmp_file_name)
            sd.play(data, fs)
            sd.wait()  # Wait until the audio is played completely
            print("Audio playback finished.")
        else:
            print(f"Failed to get audio data: {response.status_code}")    
              

if __name__ == "__main__": 
    main()

