import nibabel as nib
import SimpleITK
import matplotlib.pyplot as plt
import numpy as np

result = SimpleITK.ReadImage('./test/input/subject_004_CTA.mha')
label = SimpleITK.ReadImage(r'C:\Users\q117mt\nnunet_data\nnUNet_raw\Dataset502_aorta\labelsTR\subject_004.mha')

spacing = result.GetSpacing()
direction = result.GetDirection()
origin = result.GetOrigin()

print(spacing)
print(direction)
print(origin)

array = SimpleITK.GetArrayFromImage(result)
segmentation = SimpleITK.GetArrayFromImage(label)

print(array.shape)
print(segmentation.shape)

cropped_array = array[50:-50,100:-100,100:-100].copy()
cropped_label = segmentation[50:-50,100:-100,100:-100].copy()
print(cropped_label.shape)

image = SimpleITK.GetImageFromArray(cropped_array)
image.SetDirection(direction)
image.SetOrigin(origin)
SimpleITK.WriteImage(
        image,
        './test/subject_004_cropped.mha',
        useCompression=True,
    )

plt.figure()
plt.subplot(221)
plt.imshow(np.sum(array, axis = 1))
plt.subplot(222)
plt.imshow(np.sum(cropped_array, axis = 1))
plt.subplot(223)
plt.imshow(np.sum(segmentation, axis = 1))
plt.subplot(224)
plt.imshow(np.sum(cropped_label, axis = 1))
plt.show()
plt.figure()
plt.subplot(221)
plt.imshow(np.sum(array, axis = 2))
plt.subplot(222)
plt.imshow(np.sum(cropped_array, axis = 2))
plt.subplot(223)
plt.imshow(np.sum(segmentation, axis = 2))
plt.subplot(224)
plt.imshow(np.sum(cropped_label, axis = 2))
plt.show()