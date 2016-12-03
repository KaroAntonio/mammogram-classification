from utils.loaders import *

m_fid = 'data/exams_metadata_pilot.tsv'
cw_fid = 'data/images_crosswalk_pilot.tsv'
img_dir = 'data/pilot_images/'

dl = DataLoader(m_fid, cw_fid, img_dir, num_imgs=501)
train, test = dl.get_train_test(0.5)
bl = BatchLoader(train, test, batch_size=3)

print(dl.distribution())
