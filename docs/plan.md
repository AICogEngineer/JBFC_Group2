# Project Testing, Branching, and Implementation Plan
> Last updated 12/16/2025

### Modifying Dungeon Soup
Before we can go through phase 1 we have to modify the data. The reason being while technically organized is structed in the way that could make training difficult. As such Jose has gone out of his way to modify the folder and create a new version called "Dungeion Crawl Stone Soup Full_v2"

What he did is:
-Got rid of unique and arefact dir and sorted images to existing folder
-Create humanoid dir and moved player/base models to it as well as moster image(ie. dwarfs,elfs,halflings,humnas)
-delete player/trasform folder and moved images to monster
-create therianthriopes dir(ei. centaur,naga,merfolk,minotaur,siren)
-create a sub folder for draconian called "draconian_armor"
-added more creatures to dragons dir(hyra,drake,wyvern,xtahua)
-delete vault dir from monster and moved to monster dir
-moved scepter from weapons to staff

### Phase 1: Testing

Our goal for this phase is to reach the most optimal and accurate ML model (something in between fast and accurate) to recognize video game sprites and categorize them into a specific structure outlined by our training dataset from [Dungeon Crawl Stone Soup](https://opengameart.org/sites/default/files/Dungeon%20Crawl%20Stone%20Soup%20Full_0.zip).

The dataset will first be sorted in the most general and easiest forms to sort. We will then work backwards into making it more specific as the original set.

To accomplish that, we are going to test the following optimizers and different ML model implementations into 4 different branches. The first three branches are the following.
* `sgd`: Stephen
* `SGD+Momentum`: Lawson
* `Adam`: Jose

Since the training data is stationary, we will not be test RMSprop.
For SGDs, we will test out different learning rates. Adam is excluded, since it has a variable learning rate.

We will split the training data as close to 70/30 for training and testing if possible. We will split within each trained category, making sure there is enough data to train and test on in each category. If there is not, manual judgment and reasoning will be provided later.

Once the first 3 optimizer experiments are completed, they will be merged into the fourth branch, `Batching-normalization`.
* `Batching-normalization`
  * `SGD`
  * `SGD+Momentum`
  * `Adam`

We will start experimenting with batching normalization. Any other experiments will have their own branch, such as changing batch sizes, activation functions, or regularization.

We will need to normalize the data in the range of their RGB values from (0-255) to the continuous range of 0 - 1 using (rgbvalue - 0) / (255 - 0).
Since the dataset is highly imbalanced with many classes of varying sizes, we will also need to add in data augmentation by applying image transformations (flipping, rotating, etc). This will help us create a larger dataset to train off of, since the data we have per category is small. We will also try to add class weights as well. 

We may also use Keras Tuner later for hypertuning, and some caching to make the training process faster.

After experimenting, we will document all comparisons and findings into an `analysis.md`. The first usable model will be put into its own branch `alpha` for refinement and will be branched off of for any experimentation.

The refined trained model will be pushed to `main`.

### Phase 2: Implementing
We have not reached this phase yet. 
In this phase, we will use our trained model along with VectorDB to automatically categorize and move a different new unorganized and unlabeled dataset into the same folder structure as before.

**WIP**
