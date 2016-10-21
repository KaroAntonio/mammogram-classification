import csv

def load_tsv(fid):
	f = open(fid,'r')
	reader = csv.DictReader(f, delimiter='\t')
	data = []
	for row in reader:
		data += [(row)]
	f.close()
	return data

class DataLoader:
	def __init__(self, metadata_fid, crosswalk_fid, imgs_dir):
		self.metadata_fid = metadata_fid
		self.crosswalk_fid = crosswalk_fid
		self.imgs_dir = imgs_dir

		self.imgs = {} # super dict to store all img data indexed by img_ids
		self.exams = {} # each exam maps to images for that exam

		self.load_crosswalkdata(crosswalk_fid)
		self.load_metadata(metadata_fid)

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
	dl = DataLoader(m_fid, cw_fid, img_dir)
		


		


