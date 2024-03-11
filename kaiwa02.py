import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import time
import openai
import os
import sys
import queue
import requests
import threading
import re
import webrtcvad
from scipy.io import wavfile

# Set the API endpoint URL
API_URL = "http://127.0.0.1:5000/voice"  # Adjust the port number if needed
print(sd.query_devices())
input_device_id = int(input("Enter the desired input device ID, 0 for the default device: "))
output_device_id = int(input("Enter the desired output device ID, 0 for the default device: "))

openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.Client()

# 録音の設定
# VADの初期化
vad = webrtcvad.Vad(2)  # 2は感度のレベル
fs = 16000  # サンプリングレート
duration = 30  # 最大録音時間（秒）
silence_threshold = 0.01  # 無音の閾値
silence_duration = 0.5  # 無音の最大継続時間（秒）

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

# 録音関数
def record_audio():
    print("録音を開始します。話しかけてください。")
    with sd.InputStream(device=input_device_id, samplerate=fs, channels=1, callback=callback) as stream:
        silence_counter = 0
        recording = []
        start_time = time.time()
        voice_detected = False  # 声が検出されたかどうかのフラグ
        while True:
            if not q.empty():
                data = q.get()
                if detect_silence(data, silence_threshold):
                    if voice_detected:  # 声が検出された後に無音を検出した場合
                        silence_counter += len(data) / fs
                    # 声が検出されていない場合は無音カウンターを増やさない
                else:
                    silence_counter = 0
                    voice_detected = True  # 声が検出された
                recording.append(data)
                if silence_counter >= silence_duration:  # 無音の最大継続時間を超えたら
                    if voice_detected:  # 声が検出されていた場合のみ録音を終了
                        break
                if (time.time() - start_time) >= duration:  # 最大録音時間に達したら
                    break
        return np.concatenate(recording, axis=0) 

# VADのための関数
def frame_generator(frame_duration_ms, audio, sample_rate):
    # 確認: sample_rate と frame_duration_ms がスカラー値であることを確認
    assert isinstance(sample_rate, int), "sample_rate must be an integer value"
    assert isinstance(frame_duration_ms, int), "frame_duration_ms must be an integer value"
    
    n = int(sample_rate * frame_duration_ms / 1000)  
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_collector(sample_rate, frame_duration_ms, vad, audio):
    # フレームジェネレータからフレームを取得
    frames = frame_generator(frame_duration_ms, audio, sample_rate)
    voiced_frames = []  # 音声が含まれるフレームを保存するリスト
    for frame in frames:
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)  # 現在のフレームが音声かどうかを判定
        if is_speech:
            voiced_frames.append(frame)  # 音声フレームをリストに追加
        elif voiced_frames:
            yield b''.join(voiced_frames)  # 音声フレームをバイト列として結合して出力
            voiced_frames = []  # 音声フレームリストをリセット
    if voiced_frames:
        yield b''.join(voiced_frames)  # 最後の音声フレームがあれば出力

# 音声認識関数
def recognize_speech(audio_data):
    # 録音データを16ビットpcm形式で一時ファイルに保存
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file, audio_data, fs, subtype='PCM_16')
        tmp_file_path = tmp_file.name  # ファイルパスを保存

    # VADを使用して無音部分を検出し、音声がある部分のみを抽出
    audio = wavfile.read(tmp_file_path)[1]  # wavfile.readはタプルを返すため、[1]でオーディオデータを取得
    sample_rate = fs
    segments = vad_collector(sample_rate, 10, vad, audio)

    # 音声がある部分のみを連結
    segments_list = [np.frombuffer(segment, dtype=np.int16) for segment in segments]
    if segments_list:  # セグメントリストが空でない場合のみ連結を行う
        voiced_audio = np.concatenate(segments_list)
    else:
        print("音声が検出されませんでした。")
        return None  # または適切なエラー処理を行う

    # recognize_speech関数内
    with open(tmp_file_path, 'rb') as f:  # 'rb'は読み取り専用のバイナリモードを意味します
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ja"
        )

    # 使用後にファイルを明示的に削除
    os.remove(tmp_file_path)
    return response.text

# GPT APIを用いたテキスト生成関数
def generate_text(messages):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    # 応答テキストを取得
    return response.choices[0].message.content

# メインループ
messages = [
        {"role": "system", "content": """
        短く感情表現豊かな返答をする。（「。」「！」「？」をよく使う）
        英語も全て日本語表記で返答する。
        
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
        会話の際には、できるだけ短い文章で話してください。ただし、会話の流れを考慮して、適切な返答をしてください。"""},
        {"role": "user", "content": "こんにちは。"},
        {"role": "assistant", "content": "何でも聞いてね。"},
        {"role": "system", "content": "日本語での会話開始"},
    ]
max_messages = 20
audio_playback_queue = queue.Queue()  # 音声再生用のキュー

def play_audio(sentence, tmp_file_name):
    # WAVファイルを読み込んで再生
    data, fs = sf.read(tmp_file_name)
    sd.play(data, fs, device=output_device_id)
    sd.wait()  # 音声が完全に再生されるまで待つ
    print("Audio playback finished.")
    time.sleep(0.1)  # 各音声の間に0.1秒の間を設ける

def audio_playback_worker():
    while True:
        sentence, tmp_file_name = audio_playback_queue.get()
        if sentence is None:  # 終了シグナルのチェック
            break
        play_audio(sentence, tmp_file_name)
        audio_playback_queue.task_done()

# 音声再生用のワーカースレッドを起動
audio_thread = threading.Thread(target=audio_playback_worker)
audio_thread.start()

def request_audio(sentence):
    # APIに送信するパラメータ
    params = {
        'text': sentence,  # 読み上げるテキスト
        'speaker_id': 0,  # 話者のID
        'model_id': 5,  # 使用するモデルのID
        'speaker_name': "RinneElu",  # 話者の名前（speaker_idより優先される）
        'sdp_ratio': 0.2,  # SDP（Stochastic Duration Predictor）とDP（Duration Predictor）の混合比率
        'noise': 0.6,  # サンプルノイズの割合（ランダム性を増加させる）
        'noisew': 0.8,  # SDPノイズの割合（発音の間隔のばらつきを増加させる）
        'length': 1.0,  # 話速（1が標準）
        'language': 'JP',  # テキストの言語
        'auto_split': 'true',  # 自動でテキストを分割するかどうか
        'split_interval': 1,  # 分割した際の無音区間の長さ（秒）
        'assist_text': None,  # 補助テキスト（読み上げと似た声音・感情になりやすい）
        'assist_text_weight': 1.0,  # 補助テキストの影響の強さ
        'style': 'Neutral',  # 音声のスタイル
        'style_weight': 5.0,  # スタイルの強さ
        'reference_audio_path': None  # 参照オーディオパス（スタイルを音声ファイルで指定）
    }
    # GETリクエストを送信
    response = requests.get(API_URL, params=params)
    # リクエストが成功したかチェック
    if response.status_code == 200:
        # WAVデータを一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_name = tmp_file.name
        return tmp_file_name
    else:
        print(f"Failed to get audio data: {response.status_code}")
        return None

def main():
    global messages
    try:
        while True:
            audio_data = record_audio()
            text = recognize_speech(audio_data)
            if text:
                # ユーザーの発言をmessagesに追加
                messages.append({"role": "user", "content": text})
                response = generate_text(messages)
                print("あなた:", text)
                # AIの応答をmessagesに追加
                messages.append({"role": "assistant", "content": response})
                print("AIの応答:", response)
                if len(messages) > max_messages: # このとき、一番古いmessages（最初のsystemプロンプト）は保持する
                    messages = [messages[0]] + messages[-max_messages:] 
                # 応答を「。」で分割してキューに追加
                for sentence in re.split(r'[。！？]', response):
                    if sentence.strip():  # 空のセグメントは無視
                        tmp_file_name = request_audio(sentence)
                        if tmp_file_name:
                            audio_playback_queue.put((sentence, tmp_file_name))
            else:
                print("無音が検出されました。")
            time.sleep(0.1)  # CPUの負荷を下げるための小休止

            # キューの処理が完了するのを待つ
            audio_playback_queue.join()

    except KeyboardInterrupt:  # Ctrl+Cが押されたとき
        print("プログラムを終了します。")
        audio_playback_queue.put((None, None))  # 終了シグナルを送信
        audio_thread.join()  # ワーカースレッドの終了を待つ
        sys.exit()

if __name__ == "__main__":
    main()
