import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import sys
import torch
import parser_joint
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
import superpoint
import test_joint
import util
import commons
import datasets_ws

from model import net

######################################### SETUP #########################################
args = parser_joint.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
print(f"Arguments: {args}")
print(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = net.GeoLocalizationNet(args)
model = model.to(args.device)
# model = torch.nn.DataParallel(model)

best_model_state_dict = torch.load(join('checkpoints'))
model.load_state_dict(best_model_state_dict)
pca = None

######################################### DATASETS #########################################
test_ds = datasets_ws.BaseDataset(args, args.test_datasets_folder, args.test_dataset_name, "test")
print(f"Test set: {test_ds}")

######################################### TEST on TEST SET ###################################
test_time = datetime.now()
_, _, prediction = test_joint.test(args, test_ds, model, args.test_method, pca)


######################################### Stage two  #########################################
# Looking for more test functions in superpoint.py
error = superpoint.localization_village_SIFT(args, test_ds, model, prediction)




print(f"Finished in {str(datetime.now() - start_time)[:-7]}")
print(f"Prepare time: {str(test_time - start_time)[:-7]}")
print(f"Test time:  {str(datetime.now() - test_time)[:-7]}")