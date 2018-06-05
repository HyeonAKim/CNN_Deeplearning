# 필요한 라이브러리 호출
import os
import shutil
import random
from PIL import Image,ImageColor
from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
import pathlib
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img



def crawling_image(keyword, max_num, min_size, save_dir):
    # 구글 이미지 크롤러 함수
    # Argument
    # search_keyword : 수집하기 위한 검색 키워드 - 필수 입력 값
    # max_num   : 저장이미지 최대 수 - 필수 입력 값
    # min_size : 이미지 최소 사이즈  - None 일 경우 상관없이 수집
    # save_dir : 이미지 저장 경로  - 필수 입력값
    # 이미지 경로 기본값 :./Data/Dataset/분석명/Raw

    # ex : crawling_image('"bearing"',10,None,'..\\Dataset\\Bearing\\Raw\\dearing')

    # Return
    # "complete download imgages" 문구 출력

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

    google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=10, storage={'root_dir': save_dir})
    if min_size is None :
        google_crawler.crawl(keyword=keyword, offset=0, max_num=max_num, min_size=None, max_size=None)
    else :
        google_crawler.crawl(keyword=keyword, offset=0, max_num=max_num, min_size=(min_size, min_size), max_size=None)

    print('Complete download', keyword ,' images')

def change_image(dir, save_dir , keyword, extension='png' , change_size=(256,256), color=None, delete=0):
    # 이미지 정제 및 변환 함수
    # 이미지 사이즈 , 확장자 변경, 사이즈 변경 , 색상 변경이 가능함
    # Argument
    # dir : 일괄 변경할 이미지 폴더 경로 : 상위 폴더일 경우 하위 디렉토리의 파일 모두 변경 가능
    # save_dir : 변경이미지 저장 경로
    # keyword : 변경한 이미지 앞에 붙일 키워드
    # extension : 변경확장자 , 'png' , 'jpeg','jpg' - 기본값 png
    # change_size : 변경할 이미지 사이즈 (높이height,넓이width) - 기본값 256,256
    # color : 이미지 색상 , 'L', 'RGB'
    # delete : 기존 이미지 삭제 여부 1: 삭제, 0: 유지 - 기본값 0
    # ex : change_image('..\\Dataset\\Bearing\\Raw',savedir='..\\Dataset\\Bearing\\Preprocess',keyword='bearing', change_size=(100,100))


    # 이미지 사이즈 입력 받기
    change_size = tuple(change_size)

    # 이미지 저장경로 확인 및 생성
    imgdir = os.path.join(dir,keyword)
    savedir = os.path.join(save_dir,keyword)
    pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)

    #현재위치 파일 모두 가져온다.
    for filename in os.listdir(imgdir):
        img_dir = os.path.join(imgdir,filename)
        convert_img = convert_image(img_dir,savedir,keyword,filename,extension,change_size,color,delete)

    return convert_img


def convert_image(img_dir,savedir,keyword,filename,extension, change_size=(256,256), color=None,delete=0):
    # change_image 하위 함수
    # Argument
    # img_dir : change_image()에서 받은 이미지 경로
    # change_size : change_image()에서 받은 이미지 변경 사이즈
    # color : change_image()에서 받은 이미지 변경 컬러

    # Return
    # 변경된 이미지

    change_size= tuple(change_size)
    savedir = os.path.join(savedir, keyword + '_' + filename[:filename.find('.') + 1] + extension)

    if img_dir[img_dir.find('.') + 1:] == 'gif':
        os.remove(img_dir)
    else :
        try :
            # 이미지 열기
            img = Image.open(img_dir)
            # 이미지 사이즈 조정
            if change_size is None :
                resize_img = img
            else :
                resize_img = img.resize(change_size)
            # 이미지 컬러 변경
            if color == 'L':
                color_img = resize_img.convert("L")
            elif color is None:
                color_img = resize_img.convert("RGB")
            else :
                color_img = resize_img

            # 이미지 삭제 및 저장
            if delete == 1:
                os.remove(img_dir)
                color_img.save(savedir, format=extension, compress_level=1)
            elif delete == 0:
                color_img.save(savedir, format=extension, compress_level=1)
            else:
                print('Delete raw image is 1 , else 0(default)')
        except :
            print('cannot open ',img_dir, savedir)

    EndMessage = 'Convert Done : '+keyword
    return EndMessage

def image_folder_merge(img_dir, merge_dir, merge_keyword):
    # 이미지 폴더 병합 함수 생성
    # Argument
    # img_dir : 이미지가 모여있는 폴더 경로 , 하위 폴더들이 존재해야함
    # merge_keyword : 병합할 폴더 명 ex) 의자 : 의자들, 부서진의자, 아동용의자

    pathlib.Path(merge_dir).mkdir(parents=True, exist_ok=True)

    for dirname in os.listdir(img_dir):
        if dirname.find(merge_keyword) is not -1 :
            print(dirname)
            for filename in os.listdir(os.path.join(img_dir,dirname)):
                if filename.find(merge_keyword) is not -1:
                    filedir = os.path.join(img_dir,dirname,filename)
                    shutil.copy(filedir,merge_dir)

    print('Complete merge images like ',merge_keyword)

def generator_fake_image(img_num,save_dir,prefix_keyword,save_extension='png'):
    # keras.ImageDataGenerator() 로 생성한 fake 이미지 저장하는 함수
    # Argument
    # img_num : 생성하고자하는 이미지 갯수
    # save_dir : 이미지 저장 경로
    # prefix_keyword : 이미지 저장 파일명
    # save_extension : 이미지 저장 확장자

    datagen = ImageDataGenerator(
        rotation_range=40,  # 0 ~ 180 까지 사진 회전
        width_shift_range=0.2,  # 수직적, 수평적으로 범위를 조정
        height_shift_range=0.2,  #
        rescale=1. / 255,  # 0~1 사이의 값으로 표준화
        shear_range=0.2,  # shearing transformations :
        zoom_range=0.2,  # 랜덤하게 zoom in
        horizontal_flip=True,  # 이미지의 절반을 무작위로 수평으로 뒤집는것
        fill_mode='nearest'  # 회전 또는 너비/높이 이동 후에 나타날 수 있는 공간에 새로운 픽셀로 채우는 전략
    )

    for filename in os.listdir(save_dir):
        img = load_img(os.path.join(save_dir,filename))
        x = img_to_array(img)  # (3,100,100)
        x = x.reshape((1,) + x.shape)  # numpy array with shape(1,1,100,100)

        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=save_dir,
                                  save_prefix=prefix_keyword,
                                  save_format=save_extension):
            i += 1
            if i > img_num:
                break

def create_train_folder(image_dir, save_dir, lable_name , train_percent='0.7' ,delete=0):

    train_dir = os.path.join(save_dir,'train',lable_name)
    test_dir = os.path.join(save_dir,'test',lable_name)
    train_percent = float(train_percent)

    # 디렉토리 생성
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_dir).mkdir(parents=True, exist_ok=True)

    # 디렉토리내에 파일 섞어서 랜덤한 데이터 옮기기
    list = os.listdir(image_dir)
    random.shuffle(list)

    # 이미지 갯수 확인하고 데이터 분리 기준
    index = round(len(os.listdir(image_dir)) * train_percent)

    # 이미지 데이터 분리

    for filename in list[:index]:
        image_file_dir = os.path.join(image_dir,filename)
        shutil.copy(image_file_dir, train_dir)

    for filename in list[index:]:
        image_file_dir = os.path.join(image_dir, filename)
        shutil.copy(image_file_dir, train_dir)

    if delete == 1 :
        os.rmdir(image_dir)
        print('complete create train data folder and delete origin folder')
    else :
        print('complete create train data folder')