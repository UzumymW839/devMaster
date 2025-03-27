import pathlib
import pandas as pd

# BINAURAL DATA
#noise_dataset_path = pathlib.Path(r'/daten_pool/ohlems/BSE_data/multichannel_noise')
#file_lists_path = pathlib.Path(r'file_lists/multichannel_noise')
#file_lists_path.mkdir(exist_ok=True)
# print("generating file lists for binaural data ...")

# DIAMOND ARRAY DATA
noise_dataset_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/multichannel_noise_array')
file_lists_path = pathlib.Path(r'/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/2023_binauralspeechenhancement/file_lists/diamond_multichannel_noise_array')
file_lists_path.mkdir(exist_ok=True)
print("generating file lists for diamond array data ...")

filelist_fnames = [
    file_lists_path.joinpath(fname) for fname in [
        'training.csv',
        'validation.csv',
        'test.csv'
    ]
]

split_amounts = [0.8, 0.1, 0.1]

file_lists = [
    [], # training
    [], # validation
    [] # test
]

all_files = sorted(list(noise_dataset_path.glob('**/*.wav')))
abs_split_amounts = [len(all_files) * amount for amount in split_amounts]

print(len(all_files))
print(abs_split_amounts)


for file_idx, file_name in enumerate(all_files):

    if file_idx < abs_split_amounts[0]:
        # this file belongs to training
        df_idx = 0

    elif file_idx < (abs_split_amounts[0] + abs_split_amounts[1]):
        # this file belongs to validation
        df_idx = 1

    else:
        df_idx = 2

    file_lists[df_idx].append(file_name)

file_dataframes = [pd.DataFrame(data=file_list, columns=['file_name']) for file_list in file_lists]

# write each dataframe to the respective csv
for fname, df in zip(filelist_fnames, file_dataframes):
    df.to_csv(fname)
