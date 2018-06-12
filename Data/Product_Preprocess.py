from Data.DataGenerator import image_preprocess
import random
import os


# Raw_image_경로
raw_image_dic = '..\\Data\\Dataset\\Product\\Raw'
# 이미지 변경 경로
preprocess_image_dic = '..\\Data\\Dataset\\Product\\Preprocess'

# 1. 이미지 정제 및 변환
endmsg= image_preprocess.change_image(dir = raw_image_dic  ,
                                  save_dir = preprocess_image_dic ,
                                  keyword = 'test',
                                  extension='jpeg' ,
                                  change_size=(4000,4000),
                                  color=None,
                                  delete=0)
print(endmsg)

# 3. 이미지 폴더 병합
#
# image_preprocess.image_folder_merge(img_dir=preprocess_image_dic,
#                    merge_dir=merge_dir ,
#                    merge_keyword='bearing')


# 4. fake 이미지 생성
# img_num= 1
# image_preprocess.generator_fake_image(img_num=1,
#                                       img_dir=merge_dir,
#                                       save_dir=merge_dir,
#                                       prefix_keyword='Bearing',
#                                       save_extension='png')
#
