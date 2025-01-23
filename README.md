# Captcha Solving Model
 This is a model trained specifically to solve captcha, purpose of this is to further automate one of my side projects that needs continuous captcha solving, this was one of the best solutions that i came up with in the world of automation.

# A Guide on how to make your own Captcha Solving Model
- Start by collecting the images you need to the model to detect and learn
- put them in the DataSets folder with each folder relating to 1 class only ex. if the pictures are hats then you place them under hats folder.
- make sure all the pictures are same size or almost same size, because before training all images will be resized based on the parameters given.

# Parameters adjustment
![image](https://github.com/user-attachments/assets/2652f4a7-9841-4cf9-a434-a07fdd58b575)

- BATCH_SIZE - How many pictures to train at a time, this will be evaluated based on the VRAM on your GPU or RAM if you are training on CPU.
- IMG_HEIGHT, IMG_WIDTH - the size of the picture that the program will unify on all the images before training.
- NUM_WORKERS - # of cores that simultanously work with GPU to feed it data.
- EPOCHS - how many cycles will the program do, the program has a safety net to stop when OVERFITTING happens so just set it high and forget about it and program will stop when it degrades.


# Run and experiment!
- I hope this helps you, if you need help feel free to reach out to me!

