import os
import pandas as pd
import numpy as np
import argparse
import random
from src.utils import get_local_time, check_path

def main(args):

    if args.ENSEMBLE_FILES != None :
        file_list = sum(args.ENSEMBLE_FILES, [])
        num_models = len(file_list)
   
        if num_models < 2:
            raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")

        check_path(args.SAVE_PATH)
        
        path_list = [args.ENSEMBLE_PATH + filename for filename in file_list]
        
        # convert file paths to user and items from each outputs and save as a list
        output_list = []
        for idx, path in enumerate(path_list):
            curr_output = pd.read_csv(path)

            if idx == 0:
                output_list.append(curr_output.user)
            
            output_list.append(curr_output.item)
        
        total_item = []; temp_item = set()  # total_item: total ensembled data; temp_item: ensembled data for each user

        # get one data from each models
        for idx, x in enumerate(zip(*output_list)):  # x: user, item1, item2, ...
            if 10 - num_models < len(temp_item) < 10: # when (num_models) number of data can't be added to the temp_item
                item_to_add = list(set(x[1:]))
                random.shuffle(item_to_add)

                while item_to_add and len(temp_item) < 10:
                    curr_pick = item_to_add.pop()
                    temp_item.add(curr_pick)
                
            elif 10 - num_models >= len(temp_item):
                temp_item.update(list(set(x[1:])))

            if idx % 10 == 9:   # save and reset temp_item at the last data row for the current user
                total_item.extend(list(temp_item))
                temp_item = set()
        
        # save as csv
        curr_time = get_local_time()
        final_output = pd.concat([output_list[0], pd.DataFrame({'item':total_item})], axis=1)
        final_output.to_csv(os.path.join(args.SAVE_PATH, 'ensemble_' + curr_time + '.csv'), index=False)
        print(f'FILE SAVED AT: {args.SAVE_PATH}')

    else:
        raise ValueError("Ensemble할 Model을 적어도 2개 이상 입력해 주세요.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("--ENSEMBLE_FILES", nargs='+',required=True,
        type=lambda s: [item for item in s.split(',')],
        help='required: 앙상블할 submit 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. .csv와 같은 확장자도 입력해주세요.')

    arg('--ENSEMBLE_PATH',type=str, default='./output/',
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다.')
    
    arg('--SAVE_PATH', type=str, default='./ensembles/',
        help='optional: 앙상블할 파일을 저장할 경로를 전달합니다.')

    args = parser.parse_args()
    main(args)