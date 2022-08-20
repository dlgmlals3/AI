import cv2
import matplotlib.pyplot as plt
path = 'C:\work\AI\AISource\Data\Chapter_1_source_code\Chapter_1\data\\'
im_name = 'image.jpeg'
image = plt.imread(path + im_name)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(15, 8))
plt.subplot(1,3,1)
plt.imshow(image)
plt.axis('off')
plt.title('Image 1')

plt.subplot(1,3,2)
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Image 2')

plt.subplot(1,3,3)
plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.title('Gray Image')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()