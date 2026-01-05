Training:

Cleaning the data and lowering categories amounts.

Training with class weights.
Combatting overfitting with data augmentation and spatial dropouts.

No reshuffle before take and skip to prevent data leak.

Choosing to use RGBA over RGB.
Using architecture that fits this choice.
A larger batch size will create unstable training and lower accuracy.
(80 -> 70)
A simple architecture is better for a low-complexity image, especially with a smaller dataset. Using resnet, vggnet, or any other higher complexity model provides worse results in our testing.

Using GloablAveragePooling2D() is not for our use case, since the position of the pixel art matters in categorizing it. So, Flatten() is used.

Our goal for the CNN model is to detect 32x32 RGBA pixel art images and sort them into set categories pre-defined by the original dataset.

To put in RGB Dataset B, we took only the sprites with no background added, and then removed the white background with rembg. This makes these images as RGBA, compatible with out model.

The extra alpha channel gives us more information on the art's shape and distinguishes it away from any floor or wall tile better. 



Using the original modified dataset A, we achieved a 83% accuracy with 0.6 loss with 40 categories. Using our A+B same category dataset, we achieved 80% with 0.8 loss. With the new A+B 41 category dataset, we achieved the same 80% with 0.8 loss.

Note that by using class weights, validation loss will always be higher by a certain amount.

