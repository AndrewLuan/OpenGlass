import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import daisy
from playsound import playsound


class ImageRetrieval:
    def __init__(self, image_folder, audio_folder, mapping_file):
        self.image_folder = image_folder
        self.audio_folder = audio_folder
        self.database = {}
        self.mapping = self.load_mapping(mapping_file)
    
    def load_mapping(self, mapping_file):
        mapping = {}
        with open(mapping_file, 'r') as file:
            for line in file:
                image_name, audio_name = line.strip().split(',')
                mapping[image_name] = audio_name
        return mapping
    
    def build_database(self, method):
        self.database = method.make_samples(self.image_folder)
    
    def match_image(self, query_image_path, method, depth=10):
        query = method.make_sample(query_image_path)
        _, results = method.infer(query, self.database, depth)
        return results
    
    def visualize_matches(self, query_image_path, method, matches_results):
        method.visualize_matches(query_image_path, self.image_folder, matches_results)
    
    def play_audio_for_matches(self, matches_results):
        for image_name, _ in matches_results:
            audio_name = self.mapping.get(image_name)
            if audio_name:
                audio_path = os.path.join(self.audio_folder, audio_name)
                playsound(audio_path)
                break  # Play only the top match

class ORBMethod:
    def make_samples(self, image_folder):
        orb = cv2.ORB_create()
        samples = []
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                samples.append((image_name, des))
        return samples

    def make_sample(self, image_path):
        orb = cv2.ORB_create()
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        return image_path, des

    def infer(self, query, samples, depth):
        query_path, query_des = query
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        results = []
        for sample_path, sample_des in samples:
            matches = flann.knnMatch(query_des, sample_des, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            results.append((sample_path, len(good_matches)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        orb = cv2.ORB_create()
        img_query = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        kp_query, des_query = orb.detectAndCompute(img_query, None)
        
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        img_match = cv2.imread(top_match_image_path, cv2.IMREAD_GRAYSCALE)
        kp_match, des_match = orb.detectAndCompute(img_match, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(des_query, des_match, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        img_matches = cv2.drawMatches(img_query, kp_query, img_match, kp_match, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.show()

class ColorMethod:
    def make_samples(self, image_folder):
        samples = []
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            samples.append((image_name, hist))
        return samples

    def make_sample(self, image_path):
        img = cv2.imread(image_path)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return image_path, hist

    def infer(self, query, samples, depth):
        query_path, query_hist = query
        results = []
        for sample_path, sample_hist in samples:
            distance = cv2.compareHist(query_hist, sample_hist, cv2.HISTCMP_CORREL)
            results.append((sample_path, distance))
        results.sort(key=lambda x: x[1], reverse=True)
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        # 颜色直方图不容易可视化匹配点，通常只显示匹配结果即可
        query_path, query_hist = self.make_sample(query_image_path)
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(cv2.imread(query_image_path))
        plt.subplot(1, 2, 2)
        plt.title("Top Match")
        plt.imshow(cv2.imread(top_match_image_path))
        plt.show()

class DaisyMethod:
    def make_samples(self, image_folder):
        samples = []
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            descs = daisy(img, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=False)
            samples.append((image_name, descs.flatten()))
        return samples

    def make_sample(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        descs = daisy(img, step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=False)
        return image_path, descs.flatten()

    def infer(self, query, samples, depth):
        query_path, query_desc = query
        results = []
        for sample_path, sample_desc in samples:
            distance = np.linalg.norm(query_desc - sample_desc)
            results.append((sample_path, distance))
        results.sort(key=lambda x: x[1])
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        # Daisy特征不容易可视化匹配点，通常只显示匹配结果即可
        query_path, query_desc = self.make_sample(query_image_path)
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Top Match")
        plt.imshow(cv2.imread(top_match_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.show()

class EdgeMethod:
    def make_samples(self, image_folder):
        samples = []
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            edges = cv2.Canny(img, 100, 200)
            samples.append((image_name, edges.flatten()))
        return samples

    def make_sample(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 100, 200)
        return image_path, edges.flatten()

    def infer(self, query, samples, depth):
        query_path, query_edges = query
        results = []
        for sample_path, sample_edges in samples:
            distance = np.linalg.norm(query_edges - sample_edges)
            results.append((sample_path, distance))
        results.sort(key=lambda x: x[1])
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        # 边缘特征不容易可视化匹配点，通常只显示匹配结果即可
        query_path, query_edges = self.make_sample(query_image_path)
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Top Match")
        plt.imshow(cv2.imread(top_match_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.show()

class GaborMethod:
    def make_samples(self, image_folder):
        samples = []
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            gabor_kernels = []
            for theta in np.arange(0, np.pi, np.pi / 4):
                kernel = cv2.getGaborKernel((21, 21), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                gabor_kernels.append(cv2.filter2D(img, cv2.CV_8UC3, kernel).flatten())
            samples.append((image_name, np.concatenate(gabor_kernels)))
        return samples

    def make_sample(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gabor_kernels = []
        for theta in np.arange(0, np.pi, np.pi / 4):
            kernel = cv2.getGaborKernel((21, 21), 3.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            gabor_kernels.append(cv2.filter2D(img, cv2.CV_8UC3, kernel).flatten())
        return image_path, np.concatenate(gabor_kernels)

    def infer(self, query, samples, depth):
        query_path, query_gabor = query
        results = []
        for sample_path, sample_gabor in samples:
            distance = np.linalg.norm(query_gabor - sample_gabor)
            results.append((sample_path, distance))
        results.sort(key=lambda x: x[1])
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        # Gabor特征不容易可视化匹配点，通常只显示匹配结果即可
        query_path, query_gabor = self.make_sample(query_image_path)
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Top Match")
        plt.imshow(cv2.imread(top_match_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.show()

class HOGMethod:
    def make_samples(self, image_folder):
        samples = []
        hog = cv2.HOGDescriptor()
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            h = hog.compute(img)
            samples.append((image_name, h.flatten()))
        return samples

    def make_sample(self, image_path):
        hog = cv2.HOGDescriptor()
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h = hog.compute(img)
        return image_path, h.flatten()

    def infer(self, query, samples, depth):
        query_path, query_hog = query
        results = []
        for sample_path, sample_hog in samples:
            distance = np.linalg.norm(query_hog - sample_hog)
            results.append((sample_path, distance))
        results.sort(key=lambda x: x[1])
        return query_path, results[:depth]

    def visualize_matches(self, query_image_path, image_folder, matches_results):
        # HOG特征不容易可视化匹配点，通常只显示匹配结果即可
        query_path, query_hog = self.make_sample(query_image_path)
        top_match_image_path = os.path.join(image_folder, matches_results[0][0])
        plt.subplot(1, 2, 1)
        plt.title("Query Image")
        plt.imshow(cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title("Top Match")
        plt.imshow(cv2.imread(top_match_image_path, cv2.IMREAD_GRAYSCALE), cmap='gray')
        plt.show()

# Example usage
image_folder = 'path_to_image_database'
query_image_path = 'path_to_query_image.jpg'

# Initialize the Image Retrieval system
image_retrieval = ImageRetrieval(image_folder)

# Choose the method you want to use
# Change to ColorMethod(), DaisyMethod(), EdgeMethod(), GaborMethod(), or HOGMethod() as needed
method = ORBMethod()  
# Build the database
image_retrieval.build_database(method)

# Match the query image
matches_results = image_retrieval.match_image(query_image_path, method)

# Display the results
print("Top matches:")
for image_name, match_count in matches_results:
    print(f"Image: {image_name}, Matches: {match_count}")

# Visualize the matches
image_retrieval.visualize_matches(query_image_path, method, matches_results)
