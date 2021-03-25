# Challenge Crack The Captcha by Meritis

https://github.com/meritisgroup/challenge-crack-the-captcha-public/tree/java

## Prepare Java environment

requirement: Java >= 11

### Windows

mettre le <classifier>windows-x86_64-avx2</classifier> dans le pom.xml

### Linux 

mettre le <classifier>linux-x86_64-avx512</classifier> dans le pom.xml


## Code

change ` dataPath ` to path to data (directory train & test)

### Read Images
`ParcoursData.java` read image, split a captcha into letters

### Generate samples
`OxGenerate.java` prepare datasets of X O I

### Train network
`TrainNetwork.java` train a network with deeplearning4j

### View learning statistics
`ViewNetworkLearningData.java` launch DL4J training UI

### Use network
`UseNetwork.java` read an image and guess class with the network
