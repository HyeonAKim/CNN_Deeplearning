from Data.DataGenerator import image_preprocess
import random
import os

# [검색키워드 . 저장폴더명]
keywords = [['"block bearing"','block bearing'],['"linear bearing"','linear bearing'],['"roller bearing"','roller bearing'],['"ball bearing"','ball bearing']]

# Raw_image_경로
raw_image_dic = '..\\Data\\Dataset\\Bearing\\Raw'
# 이미지 변경 경로
preprocess_image_dic = '..\\Data\\Dataset\\Bearing\\Preprocess'
# 이미지 병합 경로
merge_dir = '..\\Data\\Dataset\\Bearing\\Bearing'
# 이미지 데이터분리 경로
train_test_dir =  '..\\Data\\Dataset\\Bearing'

# # 1. 이미지 크롤링
# for keyword in keywords:
#     print('Searching...' , keyword[0])
#     save_folder = os.path.join(raw_image_dic,keyword[1])
#     print('Save folder directory : ', save_folder)
#     image_preprocess.crawling_image(keyword=keyword[0]
#                                     , max_num=100
#                                     , min_size=100
#                                     , save_dir=save_folder)
#
# 2. 이미지 정제 및 변환

# for keyword in keywords:
#     endmsg= image_preprocess.change_image(dir = raw_image_dic  ,
#                                   save_dir = preprocess_image_dic ,
#                                   keyword = keyword[1],
#                                   extension='png' ,
#                                   change_size=(256,256),
#                                   color=None,
#                                   delete=0)
#     print(endmsg)
#
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

# 4. fake 이미지로 데이터 채우기
image_preprocess.fill_fake_image(img_dir=preprocess_image_dic, img_num=100 , chagne_num=1)

# 5. 학습이미지 데이터 분리
#
for keyword in keywords:
    img_file_folder = os.path.join(preprocess_image_dic,keyword[1])
    image_preprocess.create_train_folder(image_dir = img_file_folder,
                                         save_dir = train_test_dir,
                                         lable_name=keyword[1],
                                         train_percent='0.7',
                                         delete=0)

