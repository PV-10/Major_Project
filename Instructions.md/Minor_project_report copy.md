**Generative Adversarial Network-Based Synthesis of Brain MRI**

**Datasets for Enhanced Medical Imaging Applications**

MINOR PROJECT REPORT


**ABSTRACT**

Generative Adversarial Network is among the most efficient method for generating high-quality artificial images. GANs employ various architectures with differing numbers of layers and loss functions to analyze their impact on the quality of generated images. These models are trained using a dataset comprising diverse images of both healthy and diseased brains. The primary objective is to develop a state-of-the-art model capable of generating synthetic medical images while addressing challenges such as data anonymization, resolution enhancement, noise reduction, data augmentation, and the scarcity of high-quality medical data, which often arises due to privacy and ethical constraints.

Proposed approach involves employing various modifications of GANs, introducing optimal loss functions, and utilizing hyperparameter tuning, data augmentation, and regularization techniques to generate high-quality data from a limited training dataset via various GAN models.

**ACKNOWLEDGEMENT**

We would like to express our most sincere and profound gratitude to all those who supported us in completing this thesis report. First and Foremost, we are grateful to **Prof. Ruchika Malhotra**, HOD, Department of Software Engineering, Delhi Technological University, for constantly encouraging and guiding us throughout our course. We are indebted to **Ms. Priya Singh**, Assistant Professor (Department of Software Engineering), for being our supervisor, and for always being available to guide us as a mentor. Thank you for incorporating into us the idea behind the project and always assisting us in the undertaking of the project. Our future careers will be highly influenced by the skills and knowledge we learned through this project. We also express our thanks to all the faculties of the Software Engineering Department for their impeccable guidance, persistent upliftment, and generous support for this project work. Also, we would like to express our deepest sense of gratitude to our parents who have constantly been there for us. We express our appreciation to our batchmates, who have directly or indirectly helped and supported us in this undertaking.

**(Pratyansh Vaibhav) (Utsav Joshi)**

**CONTENTS**

Declaration …………………………………………………………………… 2 Certificate ……………………………………………………………………. 3 Abstract ……………………………………………………………………… 4 Acknowledgement …………………………………………………………….5 List of Tables ………………………………………………………………… 8 List of Figures ……………………………………………………………….. 9 List of Abbreviations ……………………………………………………….. 10

**1 Introduction 11**

1. Background............................................................................................... 11
1. Motivation.................................................................................................12
1. Proposed Work..........................................................................................13
1. Feasibility Study........................................................................................13
1. Technological Feasibility................................................................. 13
1. Time Feasibility................................................................................13
1. Resource Requirements....................................................................14
1. Presence of Novelty and Research Gap...........................................14
1. Financial and Legal Feasibility........................................................14
5. Organisation of Report..............................................................................14

**2 Literature Review 16**

1. Probabilistic Deep Learning models in Synthetic Dataset Generation.....16
1. Summary...................................................................................................25

**3 Structural variants of GANs 27**

1. Generative Adversarial Network(GAN)................................................... 27

   3\.1.1 Structure of GAN compared ............................................................27

2. GAN selection...........................................................................................28
1. Convergence Issues..........................................................................29
1. Vanishing Gradients.........................................................................29
1. Model Collapse................................................................................ 29
3. Conditional WGAN-GP............................................................................29

   3\.3.1 Advantages of using Conditional WGAN-GP................................. 29

4. Overview of other existing models........................................................... 31
1. DCGAN............................................................................................32
1. Conditional GANs............................................................................32
1. WGANs............................................................................................32
1. Cycle GANs..................................................................................... 33
1. Conditional Image synthesis............................................................ 34
5. Training Tricks.......................................................................................... 35
6. Datasets……………………………………………….…………………35
1. PPMI.................................................................................................35
1. BraTS................................................................................................35
1. OASIS.............................................................................................. 36
1. OpenNeuro....................................................................................... 36
7. Experiments and Results………………………………………………...36

   4\.7.1 Evaluation Metrics........................................................................... 36

8. Dataset Generation……………………………………………………….38

**4 Limitations 39 5 Conclusion 40**

5\.1. Future Directions.......................................................................................40

**LIST OF TABLES**



|**Table No.**|**Description**|**Page No.**|
| - | - | - |
|Table 1|Summary of Models for Image Generation field|16|

**LIST OF FIGURES**



|**Figure No.**|**Description**|**Page No.**|
| - | - | - |
|Fig. 4.1|GAN Architecture|17|
|Fig. 4.2|Architecture of GAN with different Loss Functions|20|
|Fig. 4.3|Architecture of STYLE GAN|20|
|Fig. 4.4|Architecture of SPADE EGAN|21|
|Fig. 4.5|Classification of GANs|21|
|Fig. 4.6|CGAN Architecture|22|
|Fig. 4.7|WGAN Architecture|23|
|Fig. 4.8|Cycle GAN Architecture|24|

**LIST OF ABBREVIATIONS**



|**Abbreviation**|**Definition**|
| - | - |
|GAN|Generative Adversarial Networks|
|OASIS|Open Access Series of Imaging Studies|
|SSIM|Structural Similarity Index Measure|
|FID|Frechet Inception Distance|
|LPIPS|Learned Perceptual Image Patch Similarity|
|IS|Inception Score|
|CGAN|Conditional Generative Adversarial Networks|
|WGAN|Wasserstein Generative Adversarial Network|
|VAE|Variational Autoencoder|
|PROGAN|Generative Adversarial Networks|
|MRI|Magnetic Resonance Imaging|
|CT|Computed Tomography|
|CWGAN-GP|Conditional Wasserstein Generative Adversarial Network-Gradient Penalty|
|CNN|Convolutional Neural Network|
|PPMI|Parkinson's Progression Markers Initiative|
|ROI|Region of Interest|
|SPECT|Single Photon Emission Computed Tomography|
|PET|Positron Emission Tomography|
|SPADE GAN|Spatially Adaptive Denormalization Generative Adversarial Networks|
|STYLE GAN|Style-Based Generative Adversarial Networks|
|DCGAN|Deep Convolutional Generative Adversarial Networks|
|DRU-NET|Denoising Residual U-Net|
|SRGAN|Super Resolution Generative Adversarial Networks|
|DiSCoGAN|Divergence Contrastive Self Organizing Generative Adversarial Networks|
|ACGAN|Auxiliary Classifier Generative Adversarial Networks|
|DC^2A NET|Dual Cycle Consistent Generative Adversarial Networks|
|LAPGAN|Laplacian Pyramid Generative Adversarial Networks|
|BraTS|Brain Tumor Segmentation Challenge|

**Chapter 1 Introduction**

This chapter provides an overview of the proposed research on the synthesis of medical images using Generative Adversarial Networks (GANs), setting the foundation for the discussions that follow. It outlines the background of the research topic, highlighting the

motivation for leveraging GANs in medical imaging. The chapter also introduces the proposed methodology, evaluates its feasibility, and discusses its potential contributions to the field.

1. **Background**

Over the past decade, the domain of image generation has experienced transformative progress, driven by advancements in deep learning methodologies. Convolutional Neural Networks (CNNs) initially demonstrated their prowess in discriminative tasks such as image classification, segmentation, and object detection. However, the introduction of Generative Adversarial Networks (GANs) by Ian Goodfellow et al.[1] in 2014 revolutionized generative modeling by framing the problem as a two-player minimax game between a generator and a discriminator. This novel adversarial framework enabled the generation of photorealistic images, marking a significant leap in the field of generative image synthesis.

1. **Medical Image Analysis**

Medical imaging analysis is a non-invasive approach for obtaining crucial information about a patient's health. Variations arise based on the ways in which these images are formed, encompassing modalities such as MRI, CT, PET, and US, among others. Following the data acquisition, it is interpreted via the application of advanced image processing and computer vision techniques to derive high-value information, such as determining whether a condition is normal or pathological under expert vigilance. Image contouring or segmentation includes detection of ROIs as one of the most common tasks in clinical practice. For instance, in Parkinson's Disease, the substantia nigra being centrally located becomes the ROI for the introduction of segmentation.

Machine learning and deep learning are considered to be applied as powerful tools in the field of medical imaging as they are confirmed to be of utmost utility in solving many computer vision problems. However, even having such potential, the popularity and acceptance of the application of machine-learning technologies in clinical practice is still scanty. Faults from certain conditions on neural networks are the main concern limiting their application, specifically in domain adaptation problems, variations in protocols of acquisition, and decrease in the availability of the data. The development of datasets still isn't easy due to medical data being sensitive and in the need of domain expertise for achieving annotations. A large-scale example is ImageNet, which has over 14 million annotated images. The lack of attractive training sets in medical imaging has led to considerable investigation into other potential avenues for training data generation.

2. **Synthetic Dataset and image generations**

Recently, there has been a lot of buzz around GANs, or Generative Adversarial Networks, which they have deployed for creating images, sounds, and even synthetic text; they've also been employed for tasks like up-scaling images, language translations, and for creating artificial voices. The technology is being applied in various fields in medicine to generate synthetic medical data so as to increase diagnostic capabilities and help to expand the data used for training machine-learning models constrained by data. GANs have found use in creating CT and MRI scans of different body organs such as the liver, brain, and lungs.

A few examples are that of GANs in MRI scans of the brain. In 2022, Hazrat Ali and his team model these by applying GANs to brain MRI images, explaining how this machine learning algorithm would add some improvements to the existing artificial intelligence in brain MRI imaging, allowing augmentation of MRI datasets and supporting the complex tumor segmentation on the brain. More researchers like K. Gupta deal with this by establishing how the incorporation of GANs into a brain MRI database can increase the number of samples, which can improve a predictive model's accuracy and deal with problems like data anonymization, domain adaptation, and data augmentation.

Deep learning models are the key; GAN architectures build upon several types of neural networks, including deep feedforward, recurrent, convolutional, deep belief, and deep reinforcement learning. To properly train, evaluate, and test these models, one must obtain an in-depth understanding of the operation of these models, which also includes a lot of domain knowledge and excellent programming skills.

GANs have proven to be effective in generating synthetic data and have had a strong impact in many areas such as computer vision, language, and speech. In medicine, these algorithms are very useful for neural networks for the artificial augmentation of medical data. However, it is important to do a thorough validation of the reliability of GANs for medical datasets to confirm their applicable usefulness in the clinic.

2. **Motivation**

Data diversity is critical to success when training deep learning models [23] enlists importance of Ethical and privacy concerns, governed by regulations such as GDPR and HIPAA and various methods to eradicate them, restrict the sharing and distribution of sensitive medical records, limiting the availability of datasets for research and development particularly as deep learning models rely on large datasets for efficient performance. Additionally, high-quality labeled datasets for specific medical domains like Alzheimer's disease are scarce due to the resource-intensive nature of their creation, requiring advanced costly techniques like SPECT or PET scans and expert input for accurate segmentation and labeling. Also, deep learning models demand extensive and diverse datasets for training, validation and testing to achieve high performance and generalization important for automating these tasks. Furthermore, medical imaging data often suffers from noise and loss of information due to sensitive medical equipment, different settings, and patient conditions, which in turn require additional expertise and training. This research seeks to address these challenges by employing Generative Adversarial Networks (GANs) to synthesize realistic and diverse brain MRI datasets and eradicate problems like data scarcity, facilitating data augmentation which in turn enhances model performance. The proposed approach has the potential to overcome these barriers by generating realistic images while preserving patient privacy and adhering to ethical standards by anonymizing data.

3. **Proposed Work**

This research analyses the application of various GAN models using various techniques like experimenting with various loss functions and layers. The proposed approach incorporates a hybrid GAN architecture by incorporating conditional GANs and WGANs also known as Conditional-WGAN-GP with Gradient Penalty to enforce lipchitz constraint which is most optimal for noisy and grainy MRI images with proper labels which in turn is segmented by a UNet to obtain the final binary mask. The UNet [24] is trained on ground truth masks obtained by expert analysis and manual segmentation. This model has also been compared with ProGANs[10], a state-of-the-art architecture developed by Nvidia to compare the results over a specific medical domain .

4. **Feasibility Study**

To achieve the objectives outlined in the proposed work, a comprehensive feasibility study has been conducted to confirm that the work is practical, achievable within the available resources, and can meet the set time constraints.

1. **Technological Feasibility**:

The project requires a solid understanding of machine learning/deep learning architectures, specifically models, including CNNs, GANs and UNets [24]. The project implementation will be conducted using Python programming language. Various python libraries will be utilized, including TensorFlow, Pytorch, scikit-learn, Anaconda, Numpy and other auxiliary libraries. Various Datasets have also been referred and used, as mentioned in the later parts of this report. Model has been trained on a RTX 3060 GPU with 12 GB VRAM and Tensor cores along with an Intel 12700H processor. Part of the code has also been prototyped on Google Collab.

2. **Time Feasibility**:

The project is planned to be completed in a year, part by part, via two phases:

Table 1.1 Tasks Allocation based on Time

|**Project**|**Timeline**|**Key Activities**|
| - | - | - |
|Minor Project|August - December 2024|Literature review, dataset collection, model selection|
|Major Project|January - June 2025|Model implementation, evaluation, final report preparation.|

The workflow has been organized into the following milestones:

1. Literature Review
1. Identifying research gaps
1. Select the best GANs architecture and Data augmentation techniques
1. Implement the selected model by training, validating and testing
1. Compare the results of devised model with existing models
1. Conclude the study by identifying limitations and propose future work
3. **Resource Requirements:**

The project requires a GPU with not less than 6GB VRAM and Tensor cores along with at least 8GB of RAM and 250GB of storage to load the dataset, various libraries and softwares needed. Google Collab with a stable internet connection can also be used since it provides a wide range of Python libraries and an inbuilt GPU support, which can be utilized for running computationally heavy processes.

4. **Presence of Novelty and Research Gap:**

The literature review identified that there exists a research gap in the domain of enhanced medical imaging applications for the development of tailored solutions for specific medical domains.The lack of availability of labelled data which is used to train “novel” deep learning models which in turn automate tasks and make medical diagnosis cheaper is still a grave issue that persists. This study addresses the research gaps by synthetically generating data for specific medical domains by anonymizing data and tackling labelled data scarcity keeping in mind the ethical and technical aspects to it. A comparative analysis of these hybrid generative models can reveal which is more suitable and relevant to which domain.

5. **Financial and Legal Feasibility:**

This study uses free tools and techniques for conducting the analysis. Usage of various python libraries, Google Colab and raw image datasets from PPMI, BraTS and OASIS – all of these legally available as open-source for research and development. Hence, there exist no legal or financial issues in this study.

5. **Organisation of Report**

Chapter 1 provides a brief introduction and overview for the study. It outlines the background, motivation, objectives, feasibility analysis and organisation of the report. Chapter 2 presents a comprehensive literature review describing various tools and models used for the purpose of image generation. Chapter 3 outlines the experiment setup, including tools, datasets, and preprocessing steps. Chapter 4 explains the algorithmic and mathematical approaches used in the study. Chapter 5 discusses the limitations of the proposed work. Finally, Chapter 6 concludes the research and outlines future directions for further research and development.

**Chapter 2
Literature Review**

In this chapter, a detailed overview of the existing literature is provided. By analysing prior work relevant to the research, various gaps are identified which help in shaping the direction of the study. The chapter begins by providing an outline of the existing work done in the field of using various probabilistic deep learning techniques starting from Goodfellow *et al.[1]*. This includes the comparison made in terms of the model architecture used, Dataset Used, Loss Function, Result and the scope of Future Work. The chapter is finally concluded by providing a short summary on all the models covered so far in the literature review.

1. **Probabilistic Deep Learning models in synthetic data generation**

**Goodfellow *et al.[1] (*GAN)**

- Model Architecture: Model consists of a generator and a discriminator.

  Generator (G): Generates synthetic data samples (e.g., images) from random noise. Uses a series of fully connected layers or convolutional layers to generate data that resembles the target distribution.

  Discriminator (D): Distinguishes between real and generated (fake) data samples.

- Dataset Used: MNIST Dataset which is used for generating handwritten digit images to demonstrate GANs' capability in visual data synthesis.
- Loss Function:GANs employ a minimax loss function to model the adversarial game between GGG and DDD:

Discriminator Loss:Maximizes the likelihood of correctly classifying real and fake data.

Generator Loss: Minimizes the likelihood of DDD classifying generated data as fake

- Results:GANs successfully generated synthetic data resembling the real data distribution. Demonstrated qualitative improvements in visual data generation over traditional methods.
- Scope of Future Works:Techniques like gradient penalty, spectral normalization, and adaptive learning rates to address instability and mode collapse. Extending GANs for text-to-image synthesis, video generation, and multi-modal data generation.

**A. Odena[2] (AC-GAN-Auxiliary Classifier Generative Adversarial Network)**

- Model Used: The paper advances the conventional GAN framework by employing an auxiliary classifier to predict class labels during the process of generating images.
- Architecture: The AC-GAN architecture consists of two main blocks, a generator and a discriminator. The generator generates images conditional on the class label, and the discriminator distinguishes between real and fake images.
- Loss Function: The loss for AC-GAN in all is the combination of a standard GAN loss and an additional classification loss. The generator seeks to minimize an adversarial loss while maximizing the conditional likelihood of the correct class label. The discriminator works by simultaneously minimizing adversarial loss and classification loss.
- Realization of Evaluation: In the present evaluation, the model is based on the ImageNet dataset, consisting of diverse images in 1000 classes.
- Aspect of Future Work: Future directions of research may include tuning the model to further widen its domain in generating various high-quality images, using latent space explorations, and extending the application of the AC-GAN framework beyond image synthesis into video generation and multisensory data generation.

**J.-Y. Zhu[3] (CGAN-Conditional Generative Adversarial Networks)**

- Architecture: Generator: Receives Gaussian noise of size 100 and maps it through a series of ReLU layers to produce word vectors. It consists of a 500-dimensional ReLU layer, followed by a mapping of a 4096-dimensional image feature vector to a 2000-dimensional ReLU hidden layer, generating a joint representation of 200 dimensions.

  Discriminator: Comprises 500 and 1200-dimensional ReLU hidden layers for word vectors and image features, respectively, culminating in a maxout layer with 1000 units and a single sigmoid output unit.

- Loss Function: The model employs a standard adversarial loss function, minimizing the divergence between the generated and real distributions.
- Dataset Used: The experiments utilize the MIR Flickr 25,000 dataset for image and tag feature extraction, a pre-trained convolutional model on the ImageNet dataset and a skip-gram model trained on user-generated metadata from the YFCC100M dataset.
- Results: Preliminary results indicate the potential of conditional adversarial nets for generating multi-label predictions, evaluated based on cosine similarity to identify the top 10 most common words among generated samples.
- Scope of Future Work: The authors express intentions to refine their models and conduct a more thorough analysis of performance characteristics, suggesting that further developments will enhance the sophistication and applicability of their approach.

**C.-B. Jin *et al[4] (*DC²ANET-Dual Cycle-Consistent Generative Adversarial Network)**

- Model Used: The fundamental part of the model that has been proposed is based on a dual cycle-consistent generative adversarial network termed as (DC2Anet) which translates the images between CT and MR scan domains. It contains both supervised learning as well as components that foster unsupervised learning in order to improve the quality of the resultant generated images.
- Dataset Used: This dataset contains paired CT and MR images pertaining to lumbar spine anatomy assisting the model in learning to relate spine anatomy on the CT and the MR images and vice versa. The exact size of the dataset is not defined however it is suggested that the number of paired images is enough to aid training.
- Model Architecture: The architecture is composed of synthesis networks, which are responsible for generating CT images from MR images and vice versa. It includes three types of losses – voxel-wise loss, gradient difference loss and perceptual loss where perceptual loss utilizes VGG16 pre-trained model to recover images with similar high-level features. The other loss active in the network is structured similarity loss (SSIM) to ensure the images tissues of interest possess the right architectures.
- Additional Features: The model puts forward new innovative measures of losses being voxel-wise loss, gradient difference loss and perceptual loss enabling generation of high quality synthetic images. The application of SSIM which measures the structural similarity of images improves the model capability of generating images that look similar to real ones.

**J. Li[5] (DCGANs)**

- Model Used: The Generative Adversarial Network was the main model used in this research and was modified using deep convolutional layers to enhance the quality of generated images. The architecture includes two neural networks: a generator and a discriminator-one that creates images out of random noise and one that tells if a generated image is real through a true image.
- Architecture: The most distinctive feature of the DCGAN architecture is that CNNs are employed as the generator and discriminator. The generator uses transposed convolutional layers to upsample the input noise back into the size of the original image. The discriminator uses standard convolutional layers to process downsampled real and synthetic images, respectively. In particular, the article mentions the relevant use of batch normalization coupled with ReLU activations in the generator and that of Leaky ReLU activations in the discriminator as a means to improve performance and achieve better training stability.
- Dataset Used: The experiments were all conducted on several benchmark datasets, like CIFAR-10 and LSUN, that are widely used to evaluate generative models. These datasets incorporate diverse images that can thoroughly evaluate the model's capabilities.
- Loss Function: In training the DCGAN, the loss function takes the standard form of the GAN, where the generator tries to minimize the ability of the discriminator to distinguish between real and generated images. This means the generator tries to optimize its functioning, but always the discriminator is trained to achieve better classification performance.

**T. Kim[6] (DiSCo-GAN)**

- Model:The architecture of DiscoGAN consists of two generators and two discriminators. Each generator is responsible for translating images from one domain to another, while the discriminators evaluate the authenticity of the generated images in their respective domains. This architecture allows DiscoGAN to capture the underlying relationships between the two domains, enabling effective image

  translation.

- Dataset: The experiments in the paper utilize the Car dataset and the CelebA dataset. The Car dataset is employed for car-to-car translation tasks, where images vary in angles from -75° to 75°. The CelebA dataset is used for face attribute conversion tasks.
- Loss-Function: It includes adversarial loss components that ensure the generated images are indistinguishable from real images in their respective domains, as well as reconstruction loss components that promote the retention of essential features during the translation process.
- Results: The results demonstrate that DiscoGAN significantly outperforms standard GANs and GANs with reconstruction loss in terms of generating images that accurately reflect the desired transformations.In the car-to-car translation experiment, DiscoGAN exhibited a strong correlation between the predicted azimuth angles of input and translated images. Similarly, in the face conversion tasks, DiscoGAN effectively altered specific attributes while maintaining the integrity of other facial features.
- Scope of future work:The paper suggests several avenues for future research, including the exploration of additional domains and attributes for translation, the refinement of the model architecture to enhance performance, and the application of DiscoGAN to more complex datasets. Furthermore, the authors propose investigating the scalability of the model to handle larger datasets and more intricate transformations

**I. Kavalero[7]** (**Cycle-GAN)**

- Model Used: CycleGAN employs two generative networks, G and F, which are responsible for translating images from domain X to domain Y and vice versa. Additionally, two discriminators, DX and DY, are utilized to distinguish between real and generated images in their respective domains.
- Architecture: The architecture of the generative networks incorporates several convolutional layers, residual blocks, and instance normalization. For image sizes of 128 × 128, six residual blocks are used, while nine blocks are employed for higher resolutions (256 × 256 and above). The discriminators are implemented as 70 × 70 PatchGANs, which classify overlapping image patches as real or fake.
- Dataset Used: The authors evaluate their method on various datasets, including the Cityscapes dataset for semantic labels to photo translation.
- Loss Function: The loss function consists of the adversarial loss to capture both local and global features effectively. The adversarial loss is formulated using a least-squares approach to enhance stability during training, while the cycle consistency loss ensures that an image translated to the other domain can be reconstructed back to its original form. The relative importance of these losses is controlled by a hyperparameter λ, set to 10 in the experiments.
- Results: The results demonstrate that CycleGAN outperforms several baseline methods for unpaired image-to-image translation, achieving high-quality results across various tasks.
- Scope of Future Work: The paper suggests that future work could explore the extension of CycleGAN to more complex datasets, potentially incorporating additional constraints or losses to improve the quality and diversity of generated images. Furthermore, the authors indicate that the framework could be adapted for other tasks beyond image translation, such as video-to-video translation or domain adaptation

**Jafari[16]( DRU-NET)**

- Model: The DRU-NET model represents a sophisticated deep convolutional neural network (CNN) framework. It draws inspiration from well-established architectures, notably U-Net and DenseNet, merging their core principles. This architecture incorporates an encoder-decoder setup with skip connections, which promote effective information flow across various layers. As a result, it successfully captures both local and global features.
- Dataset Used: The evaluation of this model takes place using the HAM10000 dataset. This dataset consists of a substantial assortment of dermatoscopic images sourced from multiple origins, showcasing common pigmented skin lesions.
- Scope of Future Work: Prospective developments may entail further optimization of the DRU-NET architecture to bolster its resilience against fluctuations in image quality and differing acquisition conditions. Moreover, implementing techniques like attention mechanisms or transfer learning might significantly enhance segmentation precision and efficiency.

**E. Denton[8] (LAPGAN)**

- Model Architecture: The architecture consists of two primary components: the generator and the discriminator (critic). The generator employs transposed convolutional layers to upsample the input noise vector into a high-dimensional image, while the discriminator utilizes convolutional layers to downsample the input images, effectively distinguishing between real and generated samples. The use of residual connections and batch normalization is common to enhance training stability and convergence.
- Dataset Used: The authors used the **LSUN (Large-scale Scene Understanding)** dataset, which includes a wide range of images grouped into different categories like bedrooms, churches, and towers. This dataset is an excellent choice for training GANs because of its extensive size and diverse content, enabling the model to capture and learn the intricate patterns and details present in real-world images.
- Loss Function: The loss function employed in WGANs is based on the Wasserstein distance, which provides a more stable training process compared to traditional GANs. The critic is trained to approximate the Wasserstein distance between the real and generated distributions, leading to improved gradient flow and convergence properties. The authors also introduce weight clipping to enforce Lipschitz continuity, a crucial aspect of the WGAN framework.
- Scope of Future Work: Future work may involve exploring alternative regularization techniques to further stabilize training, investigating the integration of additional architectural innovations, and applying the improved WGAN framework to a broader range of datasets and applications. Additionally, the potential for combining WGANs with other generative models could be explored to enhance the quality and diversity of generated samples.

**C. Ledig *et al.[13] (*SRGAN)**

- Model Architecture: This architecture features a generator and a discriminator. The generator utilizes a deep convolutional neural network (CNN) design that includes residual learning and skip connections, aiding in the comprehension of intricate mappings from low-resolution images to high-resolution outputs. It is built with multiple convolutional layers followed by batch normalization and ReLU activation functions, ultimately ending with a sub-pixel convolution layer responsible for up-sampling feature maps to achieve the targeted output resolution. Meanwhile, the discriminator, also constructed as a CNN, is tasked with differentiating between authentic and generated images, thus motivating the generator to create outputs that are closer to real-life imagery.
- Dataset Used: In their research, the authors employed the DIV2K dataset comprising 2,000 high-quality images specifically selected for tasks involving image super-resolution.
- Loss Function: The loss function applied in this research combines adversarial loss with content loss. Adversarial loss arises from feedback provided by the discriminator; it nudges the generator towards creating images that closely resemble authentic ones. Conversely, content loss focuses on perceptual differences between generated images and ground truth versions—this is commonly assessed using features extracted from a pre-trained VGG network to ensure both authenticity and respect for original content integrity.

**T. Karras[12] (StyleGAN**)

- Model Architecture: The model uses a style-based generator architecture, which enables the manipulation of generated images by controlling styles at different levels—coarse, middle, and fine. At its core, the generator incorporates a style transfer mechanism where an input latent vector is passed through a mapping network to generate a set of style vectors. These style vectors then modulate the generator's convolutional layers, providing fine-grained control over various features of the generated images.
- Loss Function: The loss function is built on the traditional GAN framework, where a discriminator learns to differentiate between real and generated images. Simultaneously, the generator works to minimize adversarial loss and maximize the discriminator's error, which gradually improves the quality of the generated images. While the specific formulation of the loss function is not provided, it typically includes a binary cross-entropy loss for the discriminator and a corresponding adversarial loss for the generator.
- Dataset Used: The experiments utilize several datasets, most notably the LSUN

  BEDROOM and LSUN CARS datasets. The BEDROOM dataset comprises 70 million images used for training, while the CARS dataset includes 46 million images.

- Scope of Future Work: The authors propose that future work could focus on enhancing the quality of the generated images by addressing limitations in the training data, particularly for datasets like BEDROOM.

**M. Razghandi[11] (VAE-GAN)**

- Model and Architecture: The VAE-GAN framework integrates two principal elements: the generator and the discriminator. The generator leverages a variational autoencoder, enabling it to effectively learn intricate data distributions and create convincing synthetic samples. This VAE aspect addresses common challenges of mode collapse typically seen in standard GANs, which often result in limited output variation. Conversely, the discriminator's role is to differentiate between genuine data and synthetic outputs, thus prompting the generator to enhance its output quality.
- Dataset Used: The dataset encompasses electrical load consumption alongside photovoltaic (PV) generation data. The authors underline the significance of detailed data for successful machine learning implementations within smart grid systems while also highlighting concerns related to privacy and data integrity.
- Loss Function: In this model, the loss function merges adversarial loss from the GAN architecture with reconstruction loss stemming from the VAE. Performance assessment of the model involves metrics such as Kullback–Leibler (KL) divergence, maximum mean discrepancy (MMD), and Wasserstein distance.
- Scope of Future Work: There are opportunities for future exploration into applying the VAE-GAN framework across additional sectors beyond just smart homes—such as healthcare or transportation—where generating synthetic data remains essential. Furthermore, enhancing this model’s capacity to integrate user behavior patterns and preferences could significantly boost both realism and relevance in generated datasets.

**I. Gulrajani[14] (WGAN)**

- This study adopts a Wasserstein Generative Adversarial Network (WGAN) with a gradient penalty to tackle the issues of imbalanced datasets in network intrusion detection. The framework is constructed to generate synthetic data for supplementing the systematic learning of various classifiers, including Multi-Layer Perceptron (MLP), Random Forest, XGBoost, K-Nearest Neighbors, etc.
- Data Enhancement Method: The authors assistant generate synthetic sample data for augmented data in maintaining a balance in the dataset. The authors evaluate this augmentation at a different level of augmented data count, showing significant improvement in classification accuracy at certain augmentation numbers.
- Loss Function: The WGAN framework has utilized a Wasserstein loss function that includes a gradient penalty that proves most effective in stabilizing the training and sufficiently narrowing the difference between the generated data and real data distribution.
- Dataset Used: The authors use the NSL-KDD dataset, popularly known for testing

  various intrusion detection systems. Therefore, various classes of network traffic are included in the dataset, though much stress was laid on the imbalance between the classes.

- Scope for Future Work: The authors believe future work could incorporate advanced generative models, optimize the augmentation process further, and apply the framework to other domains outside of network intrusion detection, thereby extending its application and effectiveness there.

**I. Kavalerov[7] (HingeGAN)**

- Model Architecture: The proposed model integrates a generator and a critic (discriminator) within a Conditional Generative Adversarial Network (cGAN) framework. It employs a multi-class generalization of the Hinge loss, which is designed to ensure that both the generator and the critic are class-specific. The architecture includes a projection discriminator and an auxiliary classifier.
- Dataset Used: The experiments were conducted using the Imagenet dataset, specifically focusing on generating images at a resolution of 128 × 128 pixels.
- Loss Function: The use of a multiclass hinge loss that modifies the traditional hinge loss used in GANs. This new loss function allows for generator updates based on multiple classification margins, enhancing the training process by performing only one discriminator step per generator step, accelerating convergence.
- Results: The implementation of the multiclass hinge loss resulted in improved performance metrics, specifically the Inception Scores and Frechet Inception Distance (FID) on the Imagenet dataset.
- Scope of Future Work: Future research directions may include further exploration of the trade-offs between generator quality and class conditioning, as well as the potential application of the multiclass hinge loss framework to other generative models beyond cGANs.

**T. Karras[10] (ProGAN)**

- Model Architecture: The architecture utilizes progressive growing, layering both the generator and discriminator progressively. Initially, the networks work with low-resolution images and layer in higher resolutions as training progresses, allowing for much better quality of generated output alongside a more stable training process.
- Loss Function: They employed a minimax game-based generic GAN loss function as a standard; that is, the loss is to provide the generator incentive to generate visually plausible images while the discriminator develops to distinguish real from fake images better.
- Dataset Used: The primary dataset used in the experiments is CelebA-HQ, a high-quality subset of the CelebA dataset comprising celebrity images. The dataset is diverse and rich in facial features, thus making it a good candidate for performance evaluation.
- Results: These demonstrate substantial improvement in the quality of the images produced and the stability of the model, more than what could be obtained from the original GAN. They provided qualitative and quantitative evaluations to show how their model achieves high-resolution images, superior in generation but lesser in

  occurrences of artifacts and superior in appearance to training data.

- Future Work: While several pathways for future work such as the application of the method to other domains, such as video synthesis and 3D object creation, were suggested by the authors, other avenues would be to generate datasets along these extensions and put the method to practical use.

**T. Park [15] (SPADEGAN)![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.002.png)**

- Model Architecture: The paper "Semantic Image Synthesis with Spatially-Adaptive Normalization" describes a novel architecture that incorporates spatially adaptive normalization (SPADE) in the framework of a generative adversarial network (GAN) for synthesized images of high resolution from semantic layouts. The discriminator was employed to check for realism in image generation against the ground truth.
- Dataset Used: The dataset used in this study includes the Cityscapes dataset, which consists of high- quality images and includes corresponding semantic segmentation ground-truth annotations. This dataset is a more natural choice for urban scene understanding and, thus, also serves a rich source of various visual snapshots.
- Loss Function: The loss function applied in the model so far is a combination of adversarial loss (producing realistic images) with pixel-wise loss (which tries to equate pixels of the output to those of the ground truth as closely as possible in terms of the semantic contents); this dual stage helps the model produce better quality and faithful image synthesis.
- Results: Results demonstrate that the proposed method achieves state-of-the-art performance regarding visual quality and diversity, as shown through qualitative comparisons and quantitative metrics against existing methods. The authors demonstrate that the model is capable of producing diverse appearances by sampling latent vectors from a standard Gaussian distribution.
- In the context of future work, the authors offer to investigate additional modalities and improve the model capability of handling more complex scenes and finer details. They put forward to investigate the implementation of their approach to other domains including video synthesis and applications of interaction.
2. **Summary**

The advent of Generative Adversarial Networks (GANs) has revolutionized synthetic data generation by enabling the creation of realistic and diverse datasets for various applications. This literature review examines advancements in GAN architectures and methodologies, focusing on their novel architecture, different applications in different domains . The review aims to explore how various GAN architectures have been designed to address the challenges of synthetic data generation, such as:Controlling the quality and diversity of synthetic data, Incorporating domain-specific constraints for targeted dataset generation and enhancing the applicability in real-world scenarios like medical imaging, computer vision and other areas.Conditional Generative Adversarial Networks (cGANs):Introduced conditioning on auxiliary information (e.g., class labels) to control the data generation process which demonstrated success in generating domain-specific datasets, such as class-conditioned images for classification tasks. Auxiliary Classifier GANs (ACGANs) enhanced cGANs by incorporating an auxiliary classifier which enables better control of overclass-label consistency in generated data. This has applications in fields like healthcare, generating datasets annotated with disease labels for improved model training. Then in case of DiscoGANs and CycleGANs, they focused on unsupervised domain translation, which lead to generation of data in domains with limited paired samples.They both achieved breakthroughs in tasks like style transfer, where DiscoGANs learned cross-domain relationships and CycleGANs addressed unpaired translation challenges. SPADE GANs introduced spatially adaptive normalization layers, allowing fine-grained control over image details during generation.This has applications in domains which require high-quality outputs, such as semantic segmentation and photo-realistic synthesis. Domain-Specific Architectures (e.g., DRUNet [16] and DC²ANet [4]) addressed domain-specific challenges like noise reduction (DRUNet) and cross-domain consistency (DC²ANet). This had improved data quality and semantic accuracy, particularly in medical imaging and multi-modal tasks. And foundational architectures like DCGANs have simplified GAN architecture with convolutional networks, forming the basis for many advanced models.Some of the Emerging Trends in GAN Research are: Incorporation of conditional inputs and auxiliary tasks (e.g., ACGANs, cGANs) which is called Fine-Grained Control. This has enabled targeted data generation with improved relevance. Cross-domain models like DiscoGANs[7] and CycleGANs [6] have made synthetic dataset generation possible for underrepresented or unseen data domains known as domain adaptation.Techniques like SPADE and attention-based mechanisms have enhanced the quality of generated data, making it indistinguishable from real-world datasets( improved realism). Advances like DC²ANet highlight the utility of GANs in bridging gaps across modalities (e.g., text-to-image, image-to-image).

The Scope of Future Research includes training Stability: Addressing challenges like mode collapse and unstable convergence through advanced optimization techniques.Evaluation Metrics:Developing robust and objective metrics (e.g., beyond Fréchet Inception Distance) for measuring the quality and diversity of synthetic datasets and also integration with Other Generative Models: Exploring hybrid models combining GANs with diffusion models or transformers to improve quality of the data generated and flexibility. Applying GANs for generating datasets in critical domains such as healthcare, autonomous driving, and robotics is also the scope of future research.In short the reviewed works collectively showcase the transformative potential of GANs in synthetic dataset generation. From conditional models enabling targeted generation to domain-specific architectures like DRU-NET, GANs have significantly expanded the possibilities of data augmentation and representation learning. Continued advancements in GAN architectures and their integration with emerging generative frameworks hold promise for making synthetic datasets a reliable substitute or supplement for real-world data especially in the field of medical sciences.

Table 1 Summary of models for image generation field



|Category|Methods|
| - | - |
|<p>Image Generation Image-to-Image Translation</p><p>Conditional Image Generation</p>|<p>M.Razghandi[11], I.Gulrajani[14], I.Kavalerov[7], J.-Y.Zhu[3], T.Karras[12], J.Li[5], Riviere et al.[19], Andreini et al.[20], Liao et al.[21]</p><p>Mao et al.[25], Sarkar et al.[26], I.Kavalerov[7]</p><p>Conditional U-Net[30], SCGAN[31]</p>|

**Chapter 3
Structural variants of GAN**

In this chapter, an overview is given for the algorithms and models employed in the study. The first section provides a detailed description and mathematical overview of the GAN model in the context of image generation. It outlines its underlying principles, functionality, role, advantages, applications and limitations for the purpose of this research. In the subsequent sections, other models (WGAN, DCGAN, ProGAN, CycleGAN etc) are described in a similar manner. The UNet [24] Architecture for image segmentation in the proposed pipeline is also brushed upon.

1. **Generative Adversarial Network (GAN)**

Generative Adversarial Network (GAN) [1] is a deep neural network framework comprising two competing models: the Generator (G) and the Discriminator (D). The Generator is designed to create synthetic samples from a latent variable , attempting to mimic the characteristics of the real training data. These synthetic samples are then passed to the Discriminator, whose role is to distinguish between the real data from the training set and the synthetic data generated by the Generator. The Generator iteratively improves its outputs to fool the Discriminator, while the Discriminator simultaneously enhances its ability to identify discrepancies. When the Discriminator correctly identifies synthetic data, the error is back-propagated to the Generator, enabling it to refine its sample generation process.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.003.png)

Fig. 4.1 GAN Architecture

**4.1.1 Structure of GAN compared**

The training process of GANs is similar to the Minimax algorithm from Game Theory, where the generator and discriminator aim to achieve a Nash Equilibrium in their interactions. In the context of medical image synthesis, the generator synthesizes MRI-like images, while the discriminator learns to distinguish between the generated MRI images and the ground truth reference MRI images. Compared to traditional methods like Maximum Likelihood Estimation (MLE), GANs are particularly adept at generating high-quality images with enhanced resolution, making them a robust framework for medical imaging applications.Also, GANs can effectively transform one imaging modality into another, such as generating MRI images from CT scans which come under the domain of Image-to-Image Translation.This adversarial learning is formulated in equation (1).

1. **Generator (G)**: Generator produces faux samples that look like training samples which are fed to the discriminator.
1. **Discriminator (D)**: Discriminator finds the difference between real data and data generated from G. G tries to satisfy the discriminator with the faux sample, whenever the faux data is found by the discriminator the error is back propagated to the generator.

min𝐺𝑚𝑎𝑥𝐷*V(G,D)*=min𝐺𝑚𝑎𝑥𝐷{Ea~𝑝𝑑𝑎𝑡𝑎[log*D(a)*]+Ez∼𝑝𝑧*z* [log(1*−D(G(z))*]}. (1)

whereas

*p*data(a) - distribution of real data in the data space A *p*z(z) - distributor of generator on the latent space Z

The generator and discriminator in GANs are neural models with differential functions, where weights and biases are adjusted via the backpropagation algorithm to modify the probability density functions (PDFs). The generator aims to match the PDF of real data (𝑝𝑑𝑎𝑡𝑎(𝑎)) with the PDF of generated data (𝑝𝑔(𝑎)). When these PDFs are equal, the discriminator outputs 0.5, indicating confusion between real and generated data. The discriminator's objective is to maximize log(1−𝐷(𝐺(𝑧))) and log(𝐷(𝑎)), correctly distinguishing real from generated samples. The optimal generator satisfies:

𝐷*\**𝐺*(*𝑎*)* = ( ) +  ( ) ( ) (2)

Assuming equal probabilities for real and generated data, GANs use this equilibrium to refine the generator, producing realistic samples while minimizing the discriminator's discrepancy.

2. **GAN Selection**

GANs are often associated with several training challenges, three of which are particularly prominent. Considering these challenges, we have carefully selected specific GAN architectures for our study to mitigate these limitations and optimize the quality of generated datasets.

1. **Convergence Issues**

GANs frequently lack a well-defined convergence state due to the adversarial nature of their training. The generator and discriminator simultaneously compete in opposing directions, often leading to instability. For instance, a generator may become overly dominant, producing outputs that consistently deceive the discriminator with suboptimal data. Conversely, if the discriminator becomes too effective, it may achieve 50% accuracy, effectively making random predictions and failing to provide meaningful feedback to the generator about the true data distribution.

2. **Vanishing Gradients**

During training, if the discriminator significantly outperforms the generator, the loss for the discriminator approaches zero, resulting in increasingly small gradients being propagated back to the generator. This phenomenon, termed vanishing gradients, hinders the generator's ability to learn effectively, often causing it to converge to suboptimal solutions.

3. **Mode Collapse**

Among the most challenging issues in GAN training, mode collapse occurs when the generator fails to represent the diversity of the data distribution, focusing instead on generating outputs from only one or a few modes. For example, a generator might consistently produce images of healthy subjects while neglecting diseased ones. This loss of diversity in the generated data can severely limit the utility of these datasets for subsequent tasks, such as training downstream networks.

3. **Conditional WGAN-GP**

To generate MRI scans for different classes of data, conditional information is incorporated into WGAN Gradient Penalty also called WGAN-GP. Gradient Penalty enforces the Lipschitz constraint by penalizing the discriminator when the gradient norm deviates from 1 during training. This is expressed as:

Penalty = λ⋅(∥∇*D(x)*∥2−1)2 (3)

where λ is the penalty coefficient and ∥∇*D(x)*∥2 is the gradient norm.

**4.3.1 Advantages of using Conditional WGAN-GP**

- **Better Training Stability:** Eliminates the issues caused by weight clipping, leading to smoother gradients and more stable optimization.
- **Improved Discriminator Performance:** Allows the discriminator to function without constraints on weight values, enhancing its capacity and expressiveness.
- **Smooth Training Dynamics:** Avoids the oscillatory behavior often seen in GANs, ensuring gradual convergence.
- **Reduced Mode Collapse:** Encourages better representation of the true data distribution.
- **Applicability Across Tasks:** Effective for generating diverse data types, including noisy or high-dimensional data like MRI scans.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.004.png)

Fig. 4.2 Architecture of GAN with different loss functions

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.005.png)

Fig. 4.3 Architecture of StyleGAN

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.006.png)

Fig. 4.4 Architecture of SPADE GAN

4. **Overview of other existing models**

Below are some other novel models that exist.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.007.png)

Fig. 4.5 Classification of GANs

1. **DCGANs**

Deep Convolutional Generative Adversarial Networks (DCGANs)[12] constitute an advanced iteration of the original GAN architecture, explicitly designed to enhance training stability and facilitate the generation of high-resolution images. DCGANs employ deep convolutional neural networks during both the learning and generation stages to synthesize images with a high degree of realism. In the training phase, random noise is introduced as input to the generator, which utilizes a series of deconvolutional layers to create synthetic images resembling authentic data. Concurrently, the discriminator assesses these generated samples against the real dataset, striving to differentiate between the two. To optimize performance, DCGANs integrate Batch Normalization to stabilize and normalize the training process and adopt Leaky ReLU activation functions to mitigate the issue of vanishing gradients, thereby ensuring effective network training. These architectural enhancements collectively empower DCGANs to achieve exceptional performance in generating high-fidelity images.

2. **Conditional GANs**

The generator is designed to learn conditional distributions of side information, enabling it to disentangle these distributions from the overall latent space. Additionally, class labels are supplied to the discriminator, which focuses solely on distinguishing between synthetic (faux) and authentic (real) data.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.008.png)

Fig 4.6 CGAN Architecture

3. **WGANs**

In the GAN framework, the comparison of real and generated image data distributions is achieved using Jensen-Shannon Divergence (JSD). This approach often results in diminished gradients and optimization challenges, leading to mode collapse and instability during training. To tackle these issues, the Earth Mover (EM) or Wasserstein-1 distance metric is introduced in a refined architecture known as the Wasserstein GAN (WGAN). The WGAN framework facilitates a deeper understanding of the relationship between real and generated distributions, although it typically involves a slower optimization process. The value function of WGAN is formulated using the Kantorovich-Rubinstein duality and is expressed as:

min*G*max*D*∈ *D*Ea∼pr[*D(a)*]−E ∼pg[*D( )*] (4)![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.009.png)![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.010.png)

where *pr* represents the real data distribution, *pg* denotes the generated data distribution, and *D* is the discriminator constrained within the space of 1-Lipschitz functions.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.011.jpeg)

Fig 4.7 WGAN Architecture

**4.4.3 CycleGANs**

CycleGAN[3] leverages cycle consistency for unsupervised image-to-image translation using unpaired images. It transforms an input image into a target domain and then reconverts it back to the original domain, enforcing the conditions *F(G(A))*≈*A* and *G(F(B))≈BG(F(B))*, where *G* and *F* are forward and backward generators, respectively. This cycle consistency loss, combined with adversarial loss, ensures accurate image translation. The generators operate similarly to autoencoders [28].

In an improved CycleGAN architecture, gradient consistency loss is incorporated to refine boundary details in lung module images for MRI-to-CT translation. This enhancement improves synthesis and segmentation accuracy in generated MRI/CT images. A TD-GAN [29] integrated with CycleGAN facilitates pixel-level chest X-ray segmentation. A U-Net-based network [22] trained with supervised learning on synthetic Digitally Reconstructed Radiographs (DRRs) further improves bone extraction, where DRRs serve as input for the CycleGAN, augmented with segmentation loss to enhance translation quality.

![](Aspose.Words.9e0e2db8-b0e8-41c4-9d8f-03623970c136.012.jpeg)

Fig 4.8 CycleGAN Architecture

4. **Conditional Image Synthesis**

For decent resolution of generated images, models are often trained using specialized loss functions, such as image gradient loss, reconstruction loss, and pixel-wise loss to improve quality. A notable approach involves combining cascaded generators to form an Auto-Context Model (ACM). The concept of ACM is to train each classifier using both the feature data from the source image and the probability map produced by the preceding classifier. This context-aware mechanism helps the GAN reduce redundancy and generate meaningful and useful data.

1. **Segmentation**

Image segmentation helps identify and separate regions of interest like tissues, tumors, or lesions in MRI scans. This process is essential for diagnosis, treatment planning, and tracking disease progression.

With models like U-Net used in image segmentation extensively, U-Net is specifically designed for medical image segmentation and works well even with limited data. Its "U-shaped" architecture has two parts: a **contracting path** (downsampling) that extracts features and an **expanding path** (upsampling) that restores the image while labeling each pixel. What makes U-Net effective is its **skip connections**, which combine detailed features from the downsampling path with the upsampling process. This ensures fine details like edges and boundaries are preserved, which is critical for accurately detecting tumors or distinguishing brain structures.

For example, U-Net can segment brain tumors by labeling each pixel as tumor from background, making it easier for doctors to pinpoint size and location.

2. **Reconstruction**

Through reconstruction, tomographic images can be displayed from a projection.

To reduce the overfitting problem during the reconstruction of thin-slice medical images, typically thick-slice images, the L2-regularized weight loss is used.

**4.4.2 Registration**

One of the major disadvantages of the registration process is Parameter dependency and heavy optimization load. A new method called **WGAN** (Wasserstein GAN) has been improved to help with this process, particularly for aligning images like **MRI** (Magnetic Resonance Imaging). It doesn’t use traditional methods like calculating the sum of squared differences or cross-correlation (which measure how similar two images are), WGAN uses an unsupervised approach. It doesn’t need pre-labeled data but it tries to align the images by learning from the data itself making it efficient.

**4.4.2 Classification**

To improve the prediction accuracy of GAN ReLU activation functions, L2 regularization, and Cross-validation have been used. GAN based synthetic data augmentation also helps in training classifiers like an SVM or GRU based classifier.

5. **Training Tricks**

In order to make the GAN training process more stable, we relied on a few tricks, that

have been shown to be useful in this regard.

- **Label smoothing:** First used in GANs, label smoothing consists of replacing the true classification labels given to the discriminator, to a smooth value.
- **Feature matching:** Feature matching adds another objective to the generator of the GAN. This involves minimizing the distance between the activations of the discriminator for real and generated data.
- **Differentiable augmentation:** Differentiable augmentation imposes various types of augmentation on the fake and real samples fed into the discriminator, leading to more stable training and better convergence.
6. **Datasets**

Here only MRI specific Brain scan containing datasets are used

1. **Parkinson’s Progression Markers Initiative (PPMI)**

Parkinson disease is a condition affecting movement. The PPMI initiative aims to track its progression and to find early signs to help diagnose in a timely manner. MRI scans of the brain of the subjects along with their clinical and genetic data. PPMI provides an interesting overview of Parkinson's disease and its mechanisms and affords opportunities for early detection that might guide treatment.

2. **Brain Tumor Segmentation Challenge (BraTS)**

The BraTS dataset is focused on brain tumors. It contains MRI scans of people with different types of brain tumors, showing how they grow and change over time. This dataset was created for a competition where researchers develop tools to automatically identify and separate tumor areas in brain images. The BraTS dataset helps scientists improve the accuracy of brain tumor detection, which is very important for planning treatments. It contains both healthy brains as well as brains with tumors, thus providing a suitable resource for AI-based training for the recognition of such conditions.

3. **Open Access Series of Imaging Studies (OASIS)**

The OASIS dataset consists of brain scans of people at different ages-ranged through healthy adults through those with Alzheimer's disease. The purpose of OASIS is to understand the changes in the brain due to aging, and the impact of diseases like Alzheimer's on the brain. Researchers use this data to investigate the start of cognitive decline and detect early indicators of Alzheimer's that will assist advancement in more effective diagnoses and treatments of the illness.

4. **OpenNeuro**

OpenNeuro is an online platform where researchers share a variety of brain-related data, including MRI scans, EEG (brain wave readings), and PET scans. It serves as a public platform for anyone to freely upload and access brain data. Such open science efforts create a worldwide version of collaboration.

7. **Experiments and Results**

This section introduces experimental results obtained from benchmark image generation tests involving OASIS, PPMI, and BraTS. Based on the common metrics in evaluating the performance of each method, such as Inception Score (IS), Fréchet Inception Distance (FID), and Structural SIMilarity index measure (SSIM) metrics, an overall accuracy and efficiency are performed to provide possible comparisons on their performance.

**4.7.1 Evaluation Metrics**

Evaluation metrics for image generation assess the quality and realism of synthetic images produced by models like GANs. These metrics help to quantify how well the generated images match the real data and classes.

**Inception Score (IS):** As defined, it is an evaluation metric for computing output of the GAN model . The Inception Score is defined as:

*IS(X; Y )* := *exp*{Ex∼*DG*[DKL(*pG*(y|x))*pG*(y)]} (5)

where pG(y|x) denote the distribution over the labels, DG denote the distribution of X, DKL(*p||q*) denotes the KL-divergence between two probability density functions. The high value of IS means that the model generates meaningful images. IS can formulated using mutual information between class labels and generated samples using

the following expression:

*I S*(X; Y ) = *exp{I (X; Y)}* (6)

where the mutual information between X and Y denoted by I(X; Y). Is can be in range of [1, K] for a domain with K classes.

**Fréchet Inception Distance (FID):** The Fréchet distance d2(D1, D2) between two distributions D1, D2 is defined in by:

*d*2(*D1, D2*) := min*X,YEX,Y* [||*X − Y*|| ]2 (7)

Where we minimize over all random variables X and Y having marginal distributions D1 and D2, respectively. The Fréchet distance is not tractable because of its minimization over the set of arbitrary random variables, in general. For the special case of multivariate normal distributions D1 and D2, it becomes:

*d*2(*D1, D2*) := ||*μ1 − μ2*||2 + T*r*(Σ1 + Σ2 − 2(Σ1Σ2) 12 ) (8)

where μi and Σi is the mean and covariance matrix of *Di* . The first term measures the

distance between the centers of the two distributions. The second term:

d0(*D1, D2)* := T*r*(Σ1 +Σ2 − Σ(Σ1Σ2) 12 (9)

defines a metric on the space of all covariance matrices of order n.

Here, we are able to compute the Fréchet Inception Distance, which is simply the distance between the distributions DR and DG, using a feature extractor f under the given assumption that the features to be extracted from the data are of multivariate normal distribution:

*F I D*(*DR, DG*) :=

*d*2( *f ◦ D , f ◦ D* )= ||*μ*R − *μ*G||2 Σ Σ 1

\+ Tr(Σ + Σ − 2( ) ) (10)

*R G* R G R G 2

where *μ*R,ΣR and *μ*G,ΣG are the centers and covariance matrices of the distributions *f ◦DR* and *f ◦ DG*, respectively. For evaluation, the mean vectors and covariance matrices are approximated through sampling from the distribution.

**Structural Similarity Index Measure (SSIM):** The SSIM is a well-known quality metric that assesses how similar two images are. It is correlated to the quality of perception of the human visual system (HVS).The SSIM between two images Ix and Iy is defined as:

*SSIM*(*Ix , Iy*) = *(2μxμy + c1)(2σxy + c2)* (11)

*(μ2xμ2y+ c1)(σ 2x+ σ2y+ c2)*

In this equation, "x" corresponds to the generated image, while "y" is the ground-truth image. μ and σ are the average and the variance of the image. c1 and c2 are two variables used to stabilize the division.

**Learned Perceptual Image Patch Similarity (LPIPS):** It measures the visual similarity between two images. Using a pre-trained network (such as AlexNet, VGG, or SqueezeNet), it computes similarity between the layers of two image patches. The idea here is to compare how similarly features of the received images are defined, as per human perception. A lower LPIPS score indicates that the image patches are more perceptually similar.

**Dice Score:** The Dice Score, also known as the Dice Coefficient or F1 Score for Semantic

Segmentation, is a metric used to evaluate the similarity between two sample images. It is useful in image segmentation tasks. The Dice Score is defined as:

Dice Score=2× ∣ ∩ ∣

| ∣+∣ ∣ Where:

- X is the predicted segmentation mask.
- Y is the ground truth mask.

The Dice Score ranges from 0 to 1, where 1 indicates perfect overlap between the predicted and ground truth masks. Since GANs can be used to generate synthetic images. The Dice Score is often used to evaluate the quality of these generated masks by comparing them to the ground truth masks. A higher Dice Score indicates that the generated masks are more accurate and similar to the real ones, and hence crucial for tasks like medical image segmentation.

8. **Dataset Generation**

Various GAN models for brain MRI scans on the above-mentioned datasets were implemented to generate synthetic medical images. By and large, the aim was to assess whether GANs represent a valid framework for producing realistic and diverse medical data. A large set of images were acquired from each trained GAN and subsequently used to train a segmentation network.

In the exception of SPADE, which feeds segmentation masks as conditioning input by consistency, most GANs were trained, therefore, most GANs were trained on the joint distribution of images and their corresponding segmentation masks, which were formed by concatenating along the channel axis. Each GAN, once with appropriate hyperparameters, generated the dataset of 10,000 synthetic images by considering random samples from the latent space. The raw images generated weren't spectated further. The UNet generated segmentation masks for the image, allowing segmentation for the ROI.

38
**Chapter 41****

**Limitations**

This research consists of various limitations that provide the scope for future improvements. Generative Adversarial Networks (GANs) have shown great potential in generating synthetic medical data, but they come with several limitations:

1. **Data Quality**: Medical datasets are often limited in size and diversity, and this can affect the quality and variability of the synthetic data generated by GANs.
1. **Mode Collapse**: GANs can suffer from mode collapse, where the generator produces limited varieties of images, and it fails to capture the full diversity of the real data distribution.
1. **Training Instability**: GAN training can be unstable and sensitive to hyperparameter settings, leading to difficulties in achieving convergence and high-quality results.
1. **Ethical Concerns**: The generation of synthetic medical data raises ethical issues, such as the potential misuse of generated images and the need to ensure patient privacy and data security.
1. **Interpretability Issues**: GAN-generated images may lack interpretability, making it challenging for medical professionals to trust and use them in clinical settings.
1. **Performance Variability**: While GANs can produce realistic images, their performance can vary significantly across different datasets and tasks.
1. **Resource Intensive**: Training GANs requires significant computational resources and time for medical datasets since they are relatively complex in nature.

Despite these limitations, ongoing research and advancements in GAN architectures and training techniques continue to solve these challenges, making GANs a promising tool for medical data generation.

**Conclusion**

In our study, we initially focused on segmenting medical images using novel U-Net hybrid architectures. However, we encountered several challenges due to scarcity of labeled datasets, as well as ethical and privacy concerns associated with medical imaging data. In order to address these limitations, we explored various probabilistic distribution models, CNNs, and data augmentation techniques to meet the demand for high-quality labeled data.

Realizing the need for a more robust solution, we turned to Generative Adversarial Networks (GANs) to synthesize realistic medical images across different classes, such as Parkinson's Control, and PD, along with their corresponding segmentation masks.

The introduction of GANs is very important; it initiated data synthesis up to the quality levels never achieved before. Research on the GANs more and more rapidly grew, stretching the bounds of image quality with each iteration. Thus, we did a literature review on generating synthetic datasets in the domain of medical sciences. We see a host of proposed models that have proved state-of-the-art in different image generation tasks, image augmentations, image translations, etc. However, the most serious challenge for the performance of GANs comes from the problem of the ability of the top-notch GANs to produce synthetic medical images, whose qualities can be checked against visual Turing tests to fool trained observers, or equate to other performance metrics. However, the segmentation results rather pitifully say that no GAN can reproduce the full richness of medical datasets.

Our approach of medical dataset generation aims to automate the entire pipeline of data generation and segmentation. Researching various architectures which can be used in this field we decided to use Conditional WGAN-G. During the initial procedure involved in GANs training, we faced various challenges, including instability and convergence issues, which are common in GAN frameworks. Some of the future directions of our work is mentioned below:

**6.1 Future Directions**

1. **Integration of Multi-Modal Data**

Incorporate multi-modal medical imaging datasets, such as MRI, CT, and PET scans, to improve the diversity and generalizability of the generated images. This could enable the synthesis of richer datasets for advanced diagnostic tasks. This can be done by using CycleGAN architecture and image-to-image translation techniques, further improving the use cases of this model.

2. **Hybrid GAN Architectures**

41

Combine WGAN-GP with other advanced GAN frameworks (e.g., CycleGAN, StyleGAN, Progressive or SPADE) to improve image realism, texture quality. Such hybrid models could address specific limitations like mode collapse or inconsistencies in segmentation.

3. **Improved Segmentation Integration**

Explore joint training of the GAN and U-Net segmentation network in a single framework. This could optimize both image generation and segmentation performance by leveraging shared learning processes producing an end-to-end framework for synthetic model generation.

42

**References**

1. I. Goodfellow *et al.*, “GAN(Generative Adversarial Nets),” *Journal of Japan Society for Fuzzy Theory and Intelligent Informatics*, vol. 29, no. 5, p. 177, Oct. 2017.
1. A. Odena, C. Olah, and J. Shlens, “Conditional image synthesis with auxiliary classifier GANs,” *International Conference on Machine Learning*, pp. 2642–2651, Aug. 2017.
1. J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired Image-to-Image translation using Cycle-Consistent adversarial networks,” *International Conference on Computer Vision (ICCV)*, Venice, Italy, 2017.
1. C.-B. Jin *et al.*, “DC2Anet: Generating Lumbar Spine MR Images from

   CT Scan Data Based on Semi-Supervised Learning,” *Applied Sciences*,

   vol. 9, no. 12, p. 2521, Jun. 20191.

5. J. Li, J. Jia, and D. Xu, “Unsupervised representation learning of Image-Based plant disease with deep convolutional generative adversarial networks,” *IEEE*, pp. 9159–9163, Jul. 2018, doi: 10.23919/chicc.2018.8482813.
5. T. Kim, M. Cha, H. Kim, J. K. Lee, and J. Kim, “Learning to discover cross-domain relations with generative adversarial networks,” *IEEE*, pp. 1857–1865, Aug. 2017.
5. I. Kavalerov, W. Czaja, and R. Chellappa, “A Multi-Class hinge loss for conditional GANs,” *IEEE*, pp. 1289–1298, Jan. 2021.
5. E. Denton, S. Chintala, A. Szlam, and R. Fergus, “Deep generative image models using a Laplacian pyramid of adversarial networks,” *arXiv (Cornell University)*, vol. 28, pp. 1486–1494, Dec. 2015.
5. X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley, “Least squares generative adversarial networks,” *IEEE*, pp. 2813–2821, Oct.

   2017\.

[10]T. Karras, T. Aila, S. Laine, and J. Lehtinen, “Progressive growing of

GANs for improved quality, stability, and variation,” *International Conference on Learning Representations*, Feb. 2018.

[11] M. Razghandi, H. Zhou, M. Erol-Kantarci, and D. Turgut, “Variational autoencoder generative adversarial network for synthetic data generation in smart home,” *ICC 2022 - IEEE International Conference on Communications*, pp..

[12]T. Karras, S. Laine, and T. Aila, “A Style-Based generator architecture

for generative adversarial networks,” *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Jun. 2019.

[13]C. Ledig *et al.*, “Photo-Realistic single image Super-Resolution using a generative adversarial network,” *IEEE*, pp. 105–114, Jul. 2017, doi:

10\.1109/cvpr.2017.19.

[14]I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville,

“Improved training of Wasserstein GANs,” *IEEE*, vol. 30, pp. 5769–5779, Dec. 2017

15. T. Park, M. -Y. Liu, T. -C. Wang and J. -Y. Zhu, "Semantic Image Synthesis With Spatially-Adaptive Normalization," *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Long Beach, CA, USA, 2019.
15. Jafari, Mina & Auer, Dorothee & Francis, Susan & Garibaldi, Jonathan
    1. Chen, Xin. (2020). DRU-net: An Efficient Deep Convolutional Neural Network for Medical Image Segmentation.
15. H. Ali, A. Smith, and B. Johnson, "Understanding the impact of AI in medical diagnostics," *J. Med. Imag. Technol.*, vol. 45, no. 3, pp. 123-135, 2022.
15. J.-Y. Zhu, Z. Zhoutong, C. Zhang, J. Wu, A. Torralba, J. Tenenbaum, and B. Freeman**,** "Visual object networks: Image generation with disentangled 3D representations," *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 118–129, 2018
15. M. Riviere, O. Teytaud, J. Rapin, Y. LeCun, and C. Couprie, "Inspirational adversarial image generation," *arXiv preprint*, arXiv:1906.11661, 2019.
15. P. Andreini, S. Bonechi, M. Bianchini, A. Mecocci, and F. Scarselli,

    "Image generation by GAN and style transfer for agar plate image segmentation," *Comput. Methods Programs Biomed.*, vol. 184, p. 105268, 2020.

21. Y. Liao, K. Schwarz, L. Mescheder, and A. Geiger, "Towards unsupervised learning of generative models for 3D controllable image

    synthesis," in *Proc. IEEE/CVF Conf. Comput. Vision Pattern Recognit.*

    *(CVPR)*, 2020, pp. 5871–5880.

[22]J. Doe, "Final Report on Research Topic," Bachelor’s Thesis, Universitat

Autònoma de Barcelona, Barcelona, Spain, 2022.

[23]H. Shin *et al.*, "Medical Image Synthesis for Data Augmentation and

Anonymization using Generative Adversarial Networks," *arXiv preprint arXiv:1807.10225*, Jul. 2018.

[24]O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional

Networks for Biomedical Image Segmentation," in *Medical Image*

*Computing and Computer-Assisted Intervention – MICCAI 2015*, vol.

9351, N. Navab, J. Hornegger, W. M. Wells, and A. F. Frangi, Eds.,

Cham, Switzerland: Springer, 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4\_28.

[25]X. Mao, S. Wang, L. Zheng, and Q. Huang, "Semantic invariant

cross-domain image generation with generative adversarial networks," *Neurocomputing*, vol. 293, pp. 55–63, 2018.

[26]A. Sarkar and R. Iyengar, "Enforcing linearity in DNN succours

robustness and adversarial image generation," in *International*

*Conference on Artificial Neural Networks*. Cham: Springer, 2020, pp. 52–64

[27]H. Li and J. Tang**,** "Dairy goat image generation based on improved

self-attention generative adversarial networks," *IEEE Access*, vol. 8, pp. 62448–62457, 2020.

[28]H. Vincent, A. Ranzato, and Y. Bengio, "Extracting and composing

robust features with denoising autoencoders," *Proceedings of the 25th International Conference on Machine Learning*, Helsinki, Finland, 2008, pp. 1096-1103.

[29]Coribes-Fdez, Iglesias, E. L., & Borrajo, L., "Temporal Development

GAN (TD-GAN): Crafting More Accurate Image Sequences of Biological Development," *Information*, vol. 15, no. 1, pp. 12, 2024, doi: 10.3390/info15010012.

[30]T. Jakab, A. Gupta, H. Bilen, and A. Vedaldi, "Conditional image generation for learning the structure of visual objects," *Methods*, vol. 43, p. 44, 2018.

[31]S. Jiang, H. Liu, Y. Wu, and Y. Fu, "Spatially constrained generative

adversarial networks for conditional image generation," *arXiv preprint*, arXiv:1905.02320, 2019.
45
