from pyemd import emd
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import cv2
from scipy.stats import wasserstein_distance

np.random.seed(42)

def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

def initialization(A, B, p_size):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    random_B_r = np.random.randint(p, B_h-p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w-p, [A_h, A_w])
    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding[p:A_h+p, p:A_w+p, :] = A
    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
            f[i, j] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding

def propagation(f, a, dist, A_padding, B, p_size, is_odd):
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    x = a[0]
    y = a[1]
    if is_odd:
        d_left = dist[max(x-1, 0), y]
        d_up = dist[x, max(y-1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1:
            f[x, y] = f[max(x - 1, 0), y]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2:
            f[x, y] = f[x, max(y - 1, 0)]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
    else:
        d_right = dist[min(x + 1, A_h-1), y]
        d_down = dist[x, min(y + 1, A_w-1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1:
            f[x, y] = f[min(x + 1, A_h-1), y]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2:
            f[x, y] = f[x, min(y + 1, A_w-1)]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)

def random_search(f, a, dist, A_padding, B, p_size, alpha=0.5):
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h-p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y]:
            dist[x, y] = d
            f[x, y] = b
        i += 1

def NNS(img, ref, p_size, itr):
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    f, dist, img_padding = initialization(img, ref, p_size)
    for itr in range(1, itr+1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, False)
                    random_search(f, a, dist, img_padding, ref, p_size)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, True)
                    random_search(f, a, dist, img_padding, ref, p_size)
        print("iteration: %d"%(itr))
    return f


def highlight_patch_numpy(image, top_left, patch_size, color=(255, 0, 0)):
    image_copy = image.copy()
    x, y = top_left
    width, height = patch_size

    image_copy[y, x:x+width] = color
    image_copy[y+height-1, x:x+width] = color
    image_copy[y:y+height, x] = color
    image_copy[y:y+height, x+width-1] = color
    return image_copy

def plot_matched_patches(img_a, img_b, f, patch_size, index):
    i, j = index
    b_x, b_y = f[i, j]

    img_a_highlighted = highlight_patch_numpy(img_a, (j, i), (patch_size, patch_size))
    img_b_highlighted = highlight_patch_numpy(img_b, (b_y, b_x), (patch_size, patch_size))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_a_highlighted)
    axes[0].set_title("Source Image")
    axes[0].axis('off')

    axes[1].imshow(img_b_highlighted)
    axes[1].set_title("Target Image")
    axes[1].axis('off')

    plt.show()

def emd(img, ref, f, p_size):
    distances = []
    A_h, A_w = img.shape[:2]
    
    for i in range(A_h):
        for j in range(A_w):
            patch_img = img[i:i + p_size, j:j + p_size]
            i_match, j_match = f[i, j]
            patch_ref = ref[i_match:i_match + p_size, j_match:j_match + p_size]
            
            distance = wasserstein_distance(patch_img.ravel(), patch_ref.ravel())
            distances.append(distance)
    
    mean_distance = np.mean(distances)
    return mean_distance


if __name__ == "__main__":
    img = np.array(Image.open("./cup_a.jpg"))
    ref = np.array(Image.open("./cup_a.jpg"))
    p_size = 3
    itr = 5
    start = time.time()
    f = NNS(img, ref, p_size, itr)
    end = time.time()
    print("PatchMatch Time: ", end - start)

    mean_distance = emd(img, ref, f, p_size)
    print(f'Wasserstein Distance: {mean_distance}')
