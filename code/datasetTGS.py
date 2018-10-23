from torch.utils import data
from transform import *
from config import *


def do_aug(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        c = np.random.choice(4)
        if c == 0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, 0.125)  # 0.125

        if c == 1:
            image, mask = do_horizontal_shear2(image, mask, dx=np.random.uniform(-0.05, 0.05))

        if c == 2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 7))  # 10

        if c == 3:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.1))  # 0.10
            pass

    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c == 0:
            image = do_brightness_shift(image, np.random.uniform(-0.05, +0.05))
        if c == 1:
            image = do_brightness_multiply(image, np.random.uniform(1 - 0.05, 1 + 0.05))
        if c == 2:
            image = do_gamma(image, np.random.uniform(1 - 0.05, 1 + 0.05))

    return image, mask


def simple_aug(image, mask):
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
        pass

    if np.random.rand() < 0.5:
        image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 10))

    if np.random.rand() < 0.5:
        image = do_brightness_shift(image, np.random.uniform(-0.08, +0.08))

    return image, mask

def add_depth_channels(image):
    image = np.expand_dims(image, axis=0)
    out =  image.repeat(3, axis=0)
    for row, const in enumerate(np.linspace(0, 1, 101)):
        out[1, row, :] = const
    out[2] = out[0] * out[1]
    return out

class TGSSaltDataset(data.Dataset):
    def __init__(self,
                 root_path,
                 file_list,
                 divide=False,
                 # image_size=256,
                 depth=False,
                 mode='train',
                 aug='simple',
                 shape_mode='pad',
                 image_label=False):

        self.root_path = root_path
        self.file_list = file_list
        self.mode = mode

        self.divide = divide
        self.image_size = IMG_NET_SIZE

        self.orig_image_size = (101, 101)
        self.padding_pixels = None
        self.aug = aug
        self.shape_mode = shape_mode
        self.depth = depth
        self.image_label = image_label

        """
        root_path: folder specifying files location
        file_list: list of images IDs
        is_test: whether train or test data is used (contains masks or not)

        divide: whether to divide by 255
        image_size: output image size, should be divisible by 32

        orig_image_size: original images size
        padding_pixels: placeholder for list of padding dimensions
        """

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        # Get image path
        image_folder = os.path.join(self.root_path, 'images')
        image_path = os.path.join(image_folder, file_id + '.png')

        # Get mask path
        mask_folder = os.path.join(self.root_path, 'masks')
        mask_path = os.path.join(mask_folder, file_id + '.png')

        # Load image
        image = self.__load_image(image_path)
        if self.mode in ['train', 'valid']:
            mask = self.__load_image(mask_path)
            if self.divide:
                image = image / 255.
                mask = mask / 255.

            if self.image_label:
                truth_image = 1. if np.sum(image)>0 else 0.
                truth_image = np.array(truth_image)

            # AUG
            if self.mode == 'train':
                if self.aug == 'simple':
                    image, mask = simple_aug(image, mask)
                elif self.aug == 'complex':
                    image, mask = do_aug(image, mask)
                else:
                    raise NotImplementedError

            # Depth
            if self.depth:
                image = add_depth_channels(image) # 3 * 101 * 101
                image = do_resize_3_channels(image, SIZE, SIZE)
                image = do_center_pad_3_channels(image, PAD)
                image = torch.from_numpy(image).float() # 3 * 256 * 256
                # image = image.reshape(self.image_size)
            else:
                image =do_resize(image, SIZE, SIZE)
                image = do_center_pad(image, PAD)
                image = image.reshape(self.image_size, self.image_size, 1)
                image = torch.from_numpy(image).float().permute([2, 0, 1]) # 1 * 256 * 256

            mask = do_resize(mask, SIZE, SIZE)
            mask = do_center_pad(mask, PAD)
            mask = mask.reshape(self.image_size, self.image_size, 1)
            mask = torch.from_numpy(mask).float().permute([2, 0, 1])

            if self.image_label:
                return image, mask, torch.from_numpy(truth_image).float()
            else:
                return image, mask

        else:
            if self.divide:
                image = image / 255.

            # Depth
            if self.depth:
                image = add_depth_channels(image)  # 3 * 101 * 101
                image = do_resize_3_channels(image, SIZE, SIZE)
                image = do_center_pad_3_channels(image, PAD)
                image = torch.from_numpy(image).float()  # 3 * 256 * 256
                # image = image.reshape(self.image_size)
            else:
                image = do_resize(image, SIZE, SIZE)
                image = do_center_pad(image, PAD)
                image = image.reshape(self.image_size, self.image_size, 1)
                image = torch.from_numpy(image).float().permute([2, 0, 1])  # 1 * 256 * 256

            return (image,)

    def __load_image(self, path):
        return cv2.imread(str(path), 0)
    # def set_padding(self):
    #
    #     """
    #     Compute padding borders for images based on original and specified image size.
    #     """
    #
    #     pad_floor = np.floor((np.asarray(self.image_size) - np.asarray(self.orig_image_size)) / 2.)
    #     pad_ceil = np.ceil((np.asarray(self.image_size) - np.asarray(self.orig_image_size)) / 2.)
    #
    #     self.padding_pixels = np.asarray((pad_floor[0], pad_ceil[0], pad_floor[1], pad_ceil[1])).astype(np.int32)
    #
    #     return
    #
    # def __resize_image(self, img, target_shape=(224, 224)):
    #     return cv2.resize(img, dsize=target_shape)
    #
    #
    # def __pad_image(self, img):
    #
    #     """
    #     Pad images according to border set in set_padding.
    #     Original image is centered.
    #     """
    #
    #     y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[0], self.padding_pixels[1], \
    #                                                  self.padding_pixels[2], self.padding_pixels[3]
    #
    #     img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad,
    #                              x_min_pad, x_max_pad,
    #                              cv2.BORDER_REPLICATE)
    #
    #     assert img.shape[:2] == self.image_size, '\Image after padding must have the same shape as input image.'
    #
    #     return img
    #
    # def __load_image(self, path, mask=False):
    #
    #     """
    #     Helper function for loading image.
    #     If mask is loaded, it is loaded in grayscale (, 0) parameter.
    #     """
    #
    #     if mask:
    #         img = cv2.imread(str(path), 0)
    #         # img = self.__pad_image(img)
    #     else:
    #         img = cv2.imread(str(path), 0)
    #         # img = self.__pad_image(img)
    #         # img = np.expand_dims(img, axis=-1)
    #
    #
    #
    #     # height, width = img.shape[0], img.shape[1]
    #
    #
    #     return img
    #
    # def return_padding_borders(self):
    #     """
    #     Return padding borders to easily crop the images.
    #     """
    #     return self.padding_pixels
    #
    # def de_pad(self, img): # 2 dims
    #     img = img.reshape(128, 128)
    #     y_min_pad, y_max_pad, x_min_pad, x_max_pad = self.padding_pixels[0], self.padding_pixels[1], \
    #                                                  self.padding_pixels[2], self.padding_pixels[3]
    #     return img[y_min_pad:-y_max_pad, x_min_pad:-x_max_pad]


def read_ids(path):
    with open(path, 'r') as f:
        out = []
        for line in f.readlines():
            out.append(line.strip('\n'))
        return out


########################################################################################
if __name__ == '__main__':
    data_src = '/home/simon/code/20180921/data/'
    train_path = data_src + 'train'

    dir_split_data = '../split_data/'

    tr_ids = read_ids(dir_split_data + 'train_fold_{}.txt'.format(0))
    valid_ids = read_ids(dir_split_data + 'valid_fold_{}.txt'.format(0))
    test_ids = read_ids(dir_split_data + 'test.txt')

    # Training dataset:
    dataset_train = TGSSaltDataset(train_path, tr_ids, divide=True, aug='complex', shape_mode='resize', depth=True)
    dataset_train[0]