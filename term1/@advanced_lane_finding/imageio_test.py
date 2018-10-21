import imageio
from tqdm import tqdm


def main():
	reader = imageio.get_reader('project_video.mp4')
	fps = reader.get_meta_data()['fps']

	try:
		writer = imageio.get_writer('test.mp4', fps=fps)
		for i, frame in enumerate(tqdm(reader)):
			writer.append_data(frame[:, :, 1])
	except Exception as e:
		print(e)
	finally:
		writer.close()




if __name__ == '__main__':
	main()