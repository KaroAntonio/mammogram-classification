from utils.processor  import  *

m_fid = 'data/exams_metadata_pilot.tsv'
cw_fid = 'data/images_crosswalk_pilot.tsv'
img_dir = 'data/pilot_images/'

dl = DataLoader(m_fid, cw_fid, img_dir)
