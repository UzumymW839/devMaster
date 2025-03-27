def main(path = None):
    '''Function to remove all .wav files in the Directory "path"'''

    
    if path is None:
        print('Function need a path to remove the .wav files')
        return
    
    # check if path exists
    if not os.path.exists(path):
        print('Path does not exists')
        return
    
    # print path and ask for confirmation
    print(f'Path: {path}')
    confirmation = input('Do you want to remove all .wav files in this path? (y/n): ')

    if confirmation.lower() != 'y':
        print('Operation cancelled')
        return
    else:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.wav'):
                    os.remove(os.path.join(root, file))
                    print(f'{file} removed')

if __name__ == '__main__':
    import os
    import argparse
    import pathlib

    # parse arguments
    args = argparse.ArgumentParser(description='Remove all .wav files in the Directory "path"')
    args.add_argument('-p', '--path', 
                      default='/mnt/IDMT-WORKSPACE/DATA-STORE/metzrt/Data/clean_multichannel_speech_array', 
                      type=str,
                      help='file path which shall be removed (default: None)')
    

    path = pathlib.Path(args.parse_args().path)
    
    main(path)