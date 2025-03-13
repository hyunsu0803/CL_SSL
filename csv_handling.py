from sklearn.model_selection import train_test_split
import pandas as pd
import os
from glob import glob
import pickle
import soundfile as sf
  
    
def edit_csv():
    # Whatever you want, come here and do it!
    whole_file = '/root/mydir/miyungpa/metadata/gunshot_real_whole_labeled.csv'
    df = pd.read_csv(whole_file)
    # df = df.sort_values(by='filename')
    # df = df.iloc[:, 1:]
    # whole_df = whole_df.rename(columns={'filename': 'audio_path'})
    df.to_csv(whole_file, index=True)
    

def whole_filename_into_audio_path():
    base_dir = '/root/mydir/miyungpa/gunshot_online/edge-collected-gunshot-audio/'
    train_dir_list = ['38s&ws_dot38_caliber/', 'glock_17_9mm_caliber/', 'ruger_ar_556_dot223_caliber/']
    whole_file = '/root/mydir/miyungpa/metadata/gunshot_online_whole_labeled.csv'
    
    whole_df = pd.read_csv(whole_file)
    dir_list = os.listdir(base_dir)     # '38s&ws_dot38_caliber', 'glock_17_9mm_caliber', 'ruger_ar_556_dot223_caliber', 'remington_870_12_gauge'
    
    for idx, data_row in whole_df.iterrows():
        filename = data_row['filename'] + '.wav'
        for dir_name in dir_list:
            if filename in os.listdir(os.path.join(base_dir, dir_name)):
                whole_df.at[idx, 'filename'] = os.path.join(dir_name, filename)
                break
    whole_df.to_csv(whole_file, index=True)


def remove_nonexistent_datarows():
    base_dir = '/root/mydir/miyungpa/gunshot_real/'
    whole_file = '/root/mydir/miyungpa/metadata/gunshot_real_whole_labeled.csv'
    
    whole_df = pd.read_csv(whole_file)
    # dir_list = os.listdir(base_dir)     # '38s&ws_dot38_caliber', 'glock_17_9mm_caliber', 'ruger_ar_556_dot223_caliber', 'remington_870_12_gauge'
    
    drop_list = []
    for idx, data_row in whole_df.iterrows():
        filename = data_row['filename']
        drop = True
        
        #### online
        # for dir_name in dir_list:
        #     if filename in os.listdir(os.path.join(base_dir, dir_name)):
        #         drop = False
        #         break
        
        #### real
        if filename in os.listdir(os.path.join(base_dir)):
            drop = False
        
        
        if drop:
            drop_list.append(idx)
    
    whole_df = whole_df.drop(drop_list)
    whole_df.to_csv(whole_file, index=True)


def split_train_val_test():
    # Read the whole labeled data
    whole_df = pd.read_csv('/root/mydir/miyungpa/metadata/gunshot_online_whole_labeled.csv')
    
    val_test = whole_df[whole_df['audio_path'].str.contains('remington_870_12_gauge')]
    train = whole_df[~whole_df['audio_path'].str.contains('remington_870_12_gauge')]
    
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)
    train = train.sample(frac=1).reset_index(drop=True)
    
    train.to_csv('/root/mydir/miyungpa/metadata/gunshot_online_train.csv', index=False)
    val.to_csv('/root/mydir/miyungpa/metadata/gunshot_online_val.csv', index=False)
    test.to_csv('/root/mydir/miyungpa/metadata/gunshot_online_test.csv', index=False)


def show_duration_histogram():
    import pandas as pd
    import matplotlib.pyplot as plt

    # CSV 파일 경로
    csv_file_path = '/root/mydir/miyungpa/metadata/ours_train_answer.csv'

    # CSV 파일 읽기
    df = pd.read_csv(csv_file_path)

    # duration 열의 데이터 추출
    duration_data = df['duration']
    duration_data = duration_data.to_numpy(dtype=float)
    duration_data /= 44100

    # 히스토그램 그리기
    plt.figure(figsize=(10, 6))
    plt.hist(duration_data, bins=30, edgecolor='k', alpha=0.7)
    plt.title('Histogram of Duration')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/root/mydir/miyungpa/ours_train_answer_duration_histogram.png')
    plt.show()
    
    
def count_num_gunshots():

    # metadata.csv 파일 불러오기
    df = pd.read_csv('/root/mydir/miyungpa/gunshot_online/gunshot-audio-all-metadata.csv')

    # firearm별 num_gunshots 합계 계산
    firearm_gunshots_sum = df.groupby('firearm')['num_gunshots'].sum()

    # 결과 출력
    print(firearm_gunshots_sum)
        

def count_num_wavs():
    
    wavs = glob('/root/mydir/miyungpa/gunshot_real/**/*.wav', recursive=True)
    print(len(wavs))
    
    
    
def audio_path_specify():
    
    df = pd.read_csv('/root/clssl/SSL_src/metadata/test_librispeech.csv')
    
    # 'audio_path' 열의 각 경로를 수정
    def modify_path(path):
        parts = path.split('-')  # '-'로 분리
        new_path = parts[:2] + [path]  # 앞 두 요소와 전체 경로 추가
        return '/'.join(new_path)  # '/'로 연결
    
    # apply로 각 경로에 대해 modify_path 함수 적용
    df['audio_path'] = df['audio_path'].apply(modify_path)
    
    df.to_csv('/root/clssl/SSL_src/metadata/test_librispeech.csv', index=False)



def listen_pkl():
    pkl_name = '/root/clssl/SSL_src/prepared/pkl/scl/360_16_27_28_23_18_5_28_17.pkl'

    pkl_file = open(pkl_name, 'rb')
    data_dict = pickle.load(pkl_file)   # torch tensors
    pkl_file.close()
    
    mixed = data_dict['mixed'].numpy()[5, 0]
    white_snr = data_dict['white_snr_list'][5]
    print(white_snr)

    sf.write('/root/clssl/SSL_src/prepared/5.wav', mixed.T, 16000)


listen_pkl()