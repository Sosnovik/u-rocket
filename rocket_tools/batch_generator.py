import numpy as np

def preprocess_batch(batch):
    batch /= 255
    batch -= 0.5
    return batch

def BatchGenerator(image_generator, batch_size, snr=10.0, th_max=0.5, n_images = 3):
    
    while True:
        image_list = []
        mask_list = []
        
        for i in range(batch_size):
            images, masks = image_generator.get_images(n_images=n_images)
            noise_max = np.max(images) / snr
            noise = np.random.randint(0, noise_max, size=images.shape) 
            p_noise = np.random.random(size=images.shape)
            
            th_noise = np.random.uniform(0, th_max)
            images[p_noise < th_noise] = noise[p_noise < th_noise]
            
            image_list.append(images)
            mask_list.append(masks)
            

        image_list = np.array(image_list)
        image_list = preprocess_batch(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        yield image_list, mask_list