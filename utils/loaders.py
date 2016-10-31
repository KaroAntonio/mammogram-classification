import csv, os
import dicom
import numpy as np

def load_tsv(fid):
	f = open(fid,'r')
	reader = csv.DictReader(f, delimiter='\t')
	data = []
	for row in reader:
		data += [(row)]
	f.close()
	return data

class BatchLoader:
    def __init__(self, train_set, test_set, batch_size, verbose=False):
        self.batch_size = batch_size
        self.train_ptr, self.test_ptr = [-1], [-1]

        self.train_x = self.batchify(train_set['x'], batch_size)
        self.train_y = self.batchify(train_set['y'], batch_size)
        self.test_x = self.batchify(test_set['x'], batch_size)
        self.test_y = self.batchify(test_set['y'], batch_size)

        self.n_train = len(self.train_x)//2
        self.n_test = len(self.test_x)//2
        self.num_batches = self.n_train + self.n_test

        if verbose:
            print("n train batches: {}, n test batches: {}".format(self.n_train, self.n_test))
            print("feature dimensionality: {}".format(np.array(self.train_x).shape[-1]))

    def batchify(self, data, batch_size):
        '''
        return (data*2).split_into_batches()
        '''
        double_data = np.array(data*2)
        num_batches = len(double_data)//batch_size
        abridged_data = np.array(double_data[:num_batches*batch_size])
        batches = np.split(abridged_data, num_batches)
        return batches

    def next_batch(self, ptr, x, y):
        ptr[0] = (ptr[0] + 1) % len(x)
        return x[ptr[0]] , y[ptr[0]]

    def next_train(self):
        return self.next_batch(self.train_ptr, self.train_x, self.train_y)

    def next_test(self):
        return self.next_batch(self.test_ptr, self.test_x, self.test_y)

class DataLoader:
	def __init__(self, metadata_fid, crosswalk_fid, imgs_dir, num_imgs=None):
		self.metadata_fid = metadata_fid
		self.crosswalk_fid = crosswalk_fid
		self.imgs_dir = imgs_dir
		self.num_imgs = num_imgs # default None loads all imgs

		self.imgs = {} # super dict to store all img data indexed by img_ids
		self.exams = {} # each exam maps to images for that exam

		self.load_crosswalkdata(crosswalk_fid)
		self.load_metadata(metadata_fid)
		self.load_dcm_imgs(imgs_dir, num_imgs)

	def to_xy(self):
		'''
		() -> np.array,np.array
		x: np.array of model inputs
		y: np.array of model outputs
		'''
		x = []
		y = []
		view_map = {
				'L':'cancerL',
				'R':'cancerR'
				}
		for img_id,img in self.imgs.items():
			if 'dicom' in  img:
				view = img['imageView'].strip()
				pixels =  img['dicom'].pixel_array
				print(pixels.shape)
				x += [pixels]
				if view in  view_map:
					y += [int(img[view_map[view]])]
				else: print('INVALID VIEW: {}'.format(view))
		return np.array(x), np.array(y)

	def get_train_test(self, split=0.8):
		'''
		return  two dicts in form {'x':np.array, 'y':np.array}
		where split dictates the ratio of  data  points  used for training
		the  remaining data is used for  testing
		'''
		x,y = self.to_xy()

		split_i = int(len(x)*0.8)
		train =  {
			'x':x[:split_i],
			'y':y[:split_i]
			}
		test  =  {
			'x':x[split_i:],
			'y':y[split_i:]
			}
		return  train, test


	def load_dcm_img(self, fid):
		return dicom.read_file(fid)

	def load_dcm_imgs(self,dir_path, num_imgs):
		root, _, files = list(os.walk(dir_path, topdown=True))[0]
		for name in files[:num_imgs]:
			img_id = name.replace('.dcm','.dcm.gz')
			if '.dcm.gz'in img_id:
				ds = self.load_dcm_img(os.path.join(root, name))
				self.imgs[img_id]['dicom']= ds

	def load_crosswalkdata(self, fid):
		metadata = load_tsv(fid)
		for row in metadata:
			imgs_id = row['filename']
			if imgs_id in self.imgs: print(row)

			self.imgs[imgs_id] = {}
			img = self.imgs[imgs_id]

			exam_id = row['examIndex']+'_'+row['examIndex']
			if exam_id not in self.exams:
				self.exams[exam_id] = {}
			img[exam_id] = self.exams[exam_id]
			self.exams[exam_id][imgs_id] = img

			for k in row:
				img[k] = row[k]

	def load_metadata(self, fid):
		data = load_tsv(fid)
		for row in data:
			exam_id = row['examIndex']+'_'+row['examIndex']
			for imgs_id in self.exams[exam_id]:
				img = self.exams[exam_id][imgs_id]
				for k in row:
					img[k] = row[k]

if __name__ == '__main__':
	m_fid = 'data/exams_metadata_pilot.tsv'
	cw_fid = 'data/images_crosswalk_pilot.tsv'
	img_dir = 'data/pilot_images/'
	dl = DataLoader(m_fid, cw_fid, img_dir, 10)

	train,test = dl.get_train_test(0.5)

	bl =  BatchLoader(train, test, 3)
	print(bl.next_train())
