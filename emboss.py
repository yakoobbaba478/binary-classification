import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
from os import listdir
from os.path import join,splitext,basename
from imgaug import augmenters as iaa



def img_aug_emboss(image_folder):

	image_files = os.listdir(image_folder)
	pbar = tqdm(total=len(image_files))


	for an_image in image_files:
		pbar.update(1)

		image_name_split = an_image.split('.')
		extension_name = image_name_split[1]
		base_name = image_name_split[0]
		image_path = os.path.join(image_folder, an_image)

		img_file = cv2.imread(image_path)
		h, w, c = img_file.shape
		

		seq = iaa.Sequential([
			 iaa.Emboss(strength = (0, 2.0), alpha = (0, 1.0))
			])

		transformed_img = seq.augment_image(img_file)
		transformed_image_base = base_name + '_emboss'
		transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
		img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(transformed_image_path, img)


def img_aug_sharpen(image_folder):

        image_files = os.listdir(image_folder)
        pbar = tqdm(total=len(image_files))


        for an_image in image_files:
                pbar.update(1)

                image_name_split = an_image.split('.')
                extension_name = image_name_split[1]
                base_name = image_name_split[0]
                image_path = os.path.join(image_folder, an_image)

                img_file = cv2.imread(image_path)
                h, w, c = img_file.shape
                
                seq = iaa.Sequential([
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                        ])

                transformed_img = seq.augment_image(img_file)
                transformed_image_base = base_name + '_sharpen'
                transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
                img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(transformed_image_path, img)


def img_aug_crop(image_folder):

        image_files = os.listdir(image_folder)
        pbar = tqdm(total=len(image_files))


        for an_image in image_files:
                pbar.update(1)

                image_name_split = an_image.split('.')
                extension_name = image_name_split[1]
                base_name = image_name_split[0]
                image_path = os.path.join(image_folder, an_image)

                img_file = cv2.imread(image_path)
                h, w, c = img_file.shape
                
                seq = iaa.Sequential([
                        iaa.Crop(px=(0, 16))
                        ])

                transformed_img = seq.augment_image(img_file)
                transformed_image_base = base_name + '_crop'
                transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
                img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(transformed_image_path, img)


def img_aug_contrast(image_folder):

        image_files = os.listdir(image_folder)
        pbar = tqdm(total=len(image_files))


        for an_image in image_files:
                pbar.update(1)

                image_name_split = an_image.split('.')
                extension_name = image_name_split[1]
                base_name = image_name_split[0]
                image_path = os.path.join(image_folder, an_image)

                img_file = cv2.imread(image_path)
                h, w, c = img_file.shape
                

                seq = iaa.Sequential([
                        iaa.ContrastNormalization((0.5, 1.75), per_channel=0.5)
                        ])

                transformed_img = seq.augment_image(img_file)
                transformed_image_base = base_name + '_contrast'
                transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
                img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(transformed_image_path, img)


def img_aug_fliplr(image_folder):

        image_files = os.listdir(image_folder)
        pbar = tqdm(total=len(image_files))
        for an_image in image_files:
                pbar.update(1)

                image_name_split = an_image.split('.')
                extension_name = image_name_split[1]
                base_name = image_name_split[0]
                image_path = os.path.join(image_folder, an_image)

                img_file = cv2.imread(image_path)
                h, w, c = img_file.shape
              
                seq = iaa.Sequential([
                        iaa.Fliplr(1.0)
                        ])

                transformed_img = seq.augment_image(img_file)
                transformed_image_base = base_name + '_fliplr'
                transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
                img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(transformed_image_path, img)


def img_aug_blur(image_folder):

        image_files = os.listdir(image_folder)
        pbar = tqdm(total=len(image_files))


        for an_image in image_files:
                pbar.update(1)

                image_name_split = an_image.split('.')
                extension_name = image_name_split[1]
                base_name = image_name_split[0]
                image_path = os.path.join(image_folder, an_image)

                img_file = cv2.imread(image_path)
                h, w, c = img_file.shape
                
                seq = iaa.Sequential([
                        iaa.GaussianBlur((0, 0.5))
                        ])

                transformed_img = seq.augment_image(img_file)
                transformed_image_base = base_name + '_blur'
                transformed_image_path = os.path.join(image_folder, transformed_image_base + '.' +extension_name)
                img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(transformed_image_path, img)

if __name__=="__main__":
	mode = sys.argv[1]
	if(mode == "emboss"):
		image_folder=sys.argv[2]
		img_aug_emboss(image_folder)

	elif(mode == "sharpen"):
                image_folder=sys.argv[2]
                img_aug_sharpen(image_folder)

	elif(mode == "crop"):
                image_folder=sys.argv[2]
                img_aug_crop(image_folder)
	
	elif(mode == "contrast"):
                image_folder=sys.argv[2]
                img_aug_contrast(image_folder)
	
	elif(mode == "fliplr"):
                image_folder=sys.argv[2]
                img_aug_blur(image_folder)
	
	elif(mode == "blur"):
                image_folder=sys.argv[2]
                img_aug_emboss(image_folder)

	
