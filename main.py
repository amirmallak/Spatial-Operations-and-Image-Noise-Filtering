from .core_processing import *


def main():
    lena = cv2.imread(r'Images\lena.tif')
    lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # 1 ----------------------------------------------------------
    print('\n--- Part 1: Adding Salt and Pepper Noise (Low) ---')
    # add salt and pepper noise - low
    p = 0.01
    radius = 1
    std = 0.8
    lena_sp_low = add_salt_pepper_noise(lena_gray, p)  # Add low noise

    # Time consuming, so no need to calculate again
    # lena_median_clean_1 = clean_image_median_filter(lena_sp_low, radius)
    # lena_mean_clean_1 = clean_image_mean_filter(lena_sp_low, radius, std)
    # lena_bilateral_clean_1 = clean_image_bilateral_filter(lena_sp_low, radius, std, std)
    # cv2.imwrite('lena_median_clean_1.jpg', lena_median_clean_1)
    # cv2.imwrite('lena_mean_clean_1.jpg', lena_mean_clean_1)
    # cv2.imwrite('lena_bilateral_clean_1.jpg', lena_bilateral_clean_1)

    lena_median_clean_1 = cv2.imread('lena_median_clean_1.jpg')
    lena_mean_clean_1 = cv2.imread('lena_mean_clean_1.jpg')
    lena_bilateral_clean_1 = cv2.imread('lena_bilateral_clean_1.jpg')

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_low, cmap='gray', vmin=0, vmax=255)
    plt.title('Salt and Pepper - low noise')
    plt.subplot(2, 3, 4)
    plt.imshow(lena_median_clean_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Median Filter')
    plt.subplot(2, 3, 5)
    plt.imshow(lena_mean_clean_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Mean Filter')
    plt.subplot(2, 3, 6)
    plt.imshow(lena_bilateral_clean_1, cmap='gray', vmin=0, vmax=255)
    plt.title('Bilateral Filter')

    # print(f'Bilateral distance: {np.sum(lena_gray - cv2.cvtColor(lena_bilateral_clean_1, cv2.COLOR_BGR2GRAY))
    # * 1e-5:.2f}')
    # print(f'Mean distance: {np.sum(lena_gray - cv2.cvtColor(lena_mean_clean_1, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')
    # print(f'Median distance: {np.sum(lena_gray - cv2.cvtColor(lena_median_clean_1, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')

    print('Conclusions, results (by order):\n1. Median Filter\n2. Bilateral Filter \n3. Mean Filter\n')

    # 2 ----------------------------------------------------------
    print('--- Part 2: Adding Salt and Pepper Noise (High) ---')
    # add salt and pepper noise - high
    p = 0.05
    radius = 1
    std = 0.8
    lena_sp_high = add_salt_pepper_noise(lena_gray, p)

    # Time consuming, so no need to calculate again
    # lena_median_clean_2 = clean_image_median_filter(lena_sp_high, radius)
    # lena_mean_clean_2 = clean_image_mean_filter(lena_sp_high, radius, std)
    # lena_bilateral_clean_2 = clean_image_bilateral_filter(lena_sp_high, radius, std, std)
    # cv2.imwrite('lena_median_clean_2.jpg', lena_median_clean_2)
    # cv2.imwrite('lena_mean_clean_2.jpg', lena_mean_clean_2)
    # cv2.imwrite('lena_bilateral_clean_2.jpg', lena_bilateral_clean_2)

    lena_median_clean_2 = cv2.imread('lena_median_clean_2.jpg')
    lena_mean_clean_2 = cv2.imread('lena_mean_clean_2.jpg')
    lena_bilateral_clean_2 = cv2.imread('lena_bilateral_clean_2.jpg')

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateral
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.subplot(2, 3, 2)
    plt.imshow(lena_sp_high, cmap='gray', vmin=0, vmax=255)
    plt.title('Salt and Pepper - high noise')
    plt.subplot(2, 3, 4)
    plt.imshow(lena_median_clean_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Median Filter')
    plt.subplot(2, 3, 5)
    plt.imshow(lena_mean_clean_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Mean Filter')
    plt.subplot(2, 3, 6)
    plt.imshow(lena_bilateral_clean_2, cmap='gray', vmin=0, vmax=255)
    plt.title('Bilateral Filter')

    # print(f'Bilateral distance: {np.sum(lena_gray - cv2.cvtColor(lena_bilateral_clean_2, cv2.COLOR_BGR2GRAY))
    # * 1e-5:.2f}')
    # print(f'Mean distance: {np.sum(lena_gray - cv2.cvtColor(lena_mean_clean_2, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')
    # print(f'Median distance: {np.sum(lena_gray - cv2.cvtColor(lena_median_clean_2, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')

    print('Conclusions, results (by order):\n1. Median Filter\n2. Bilateral Filter \n3. Mean Filter\n')

    # 3 ----------------------------------------------------------
    print('--- Part 3: Adding Gaussian Noise (Low) ---')
    # add gaussian noise - low
    radius = 1
    std_noise = 20
    std = 0.8
    lena_gaussian = add_gaussian_noise(lena_gray, std_noise)

    # Time consuming, so no need to calculate again
    # lena_median_clean_3 = clean_image_median_filter(lena_gaussian, radius)
    # lena_mean_clean_3 = clean_image_mean_filter(lena_gaussian, radius, std)
    # lena_bilateral_clean_3 = clean_image_bilateral_filter(lena_gaussian, radius, std, std)
    # cv2.imwrite('lena_median_clean_3.jpg', lena_median_clean_3)
    # cv2.imwrite('lena_mean_clean_3.jpg', lena_mean_clean_3)
    # cv2.imwrite('lena_bilateral_clean_3.jpg', lena_bilateral_clean_3)

    lena_median_clean_3 = cv2.imread('lena_median_clean_3.jpg')
    lena_mean_clean_3 = cv2.imread('lena_mean_clean_3.jpg')
    lena_bilateral_clean_3 = cv2.imread('lena_bilateral_clean_3.jpg')

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateral
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title('Gaussian Noise - low')
    plt.subplot(2, 3, 4)
    plt.imshow(lena_median_clean_3, cmap='gray', vmin=0, vmax=255)
    plt.title('Median Filter')
    plt.subplot(2, 3, 5)
    plt.imshow(lena_mean_clean_3, cmap='gray', vmin=0, vmax=255)
    plt.title('Mean Filter')
    plt.subplot(2, 3, 6)
    plt.imshow(lena_bilateral_clean_3, cmap='gray', vmin=0, vmax=255)
    plt.title('Bilateral Filter')

    # print(f'Bilateral distance: {np.sum(lena_gray - cv2.cvtColor(lena_bilateral_clean_3, cv2.COLOR_BGR2GRAY))
    # * 1e-5:.2f}')
    # print(f'Mean distance: {np.sum(lena_gray - cv2.cvtColor(lena_mean_clean_3, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')
    # print(f'Median distance: {np.sum(lena_gray - cv2.cvtColor(lena_median_clean_3, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')

    print('Conclusions, results (by order):\n1. Mean Filter \n2. Median Filter \n3. Bilateral Filter\n')

    # 4 ----------------------------------------------------------
    print('--- Part 4: Adding Gaussian Noise (High) ---')
    # add gaussian noise - high
    radius = 1
    std_noise = 40
    std = 0.8
    lena_gaussian = add_gaussian_noise(lena_gray, std_noise)

    # Time consuming, so no need to calculate again
    # lena_median_clean_4 = clean_image_median_filter(lena_gaussian, radius)
    # lena_mean_clean_4 = clean_image_mean_filter(lena_gaussian, radius, std)
    # lena_bilateral_clean_4 = clean_image_bilateral_filter(lena_gaussian, radius, std, std)
    # cv2.imwrite('lena_median_clean_4.jpg', lena_median_clean_4)
    # cv2.imwrite('lena_mean_clean_4.jpg', lena_mean_clean_4)
    # cv2.imwrite('lena_bilateral_clean_4.jpg', lena_bilateral_clean_4)

    lena_median_clean_4 = cv2.imread('lena_median_clean_4.jpg')
    lena_mean_clean_4 = cv2.imread('lena_mean_clean_4.jpg')
    lena_bilateral_clean_4 = cv2.imread('lena_bilateral_clean_4.jpg')

    # add parameters to functions cleanImageMedian, cleanImageMean, bilateralFilt
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(lena_gray, cmap='gray', vmin=0, vmax=255)
    plt.title('Original')
    plt.subplot(2, 3, 2)
    plt.imshow(lena_gaussian, cmap='gray', vmin=0, vmax=255)
    plt.title('Gaussian Noise - High')
    plt.subplot(2, 3, 4)
    plt.imshow(lena_median_clean_4, cmap='gray', vmin=0, vmax=255)
    plt.title('Median Filter')
    plt.subplot(2, 3, 5)
    plt.imshow(lena_mean_clean_4, cmap='gray', vmin=0, vmax=255)
    plt.title('Mean Filter')
    plt.subplot(2, 3, 6)
    plt.imshow(lena_bilateral_clean_4, cmap='gray', vmin=0, vmax=255)
    plt.title('Bilateral Filter')

    # print(f'Bilateral distance: {np.sum(lena_gray - cv2.cvtColor(lena_bilateral_clean_4, cv2.COLOR_BGR2GRAY))
    # * 1e-5:.2f}')
    # print(f'Mean distance: {np.sum(lena_gray - cv2.cvtColor(lena_mean_clean_4, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')
    # print(f'Median distance: {np.sum(lena_gray - cv2.cvtColor(lena_median_clean_4, cv2.COLOR_BGR2GRAY)) * 1e-5:.2f}')

    print('Conclusions, results (by order):\n1. Mean Filter \n2. Median Filter \n3. Bilateral Filter\n')

    plt.show()


if __name__ == "__main__":
    main()
