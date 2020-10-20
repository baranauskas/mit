mkdir dot

:: ------------------------------------------------ AUDIOLOGY 

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/audiology.arff -C 0.25 -M 2 -g > dot/j48_audiology.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/audiology.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_audiology.dot



:: ------------------------------------------------ BREAST CANCER 

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/breast-cancer.arff -C 0.25 -M 2 -g > dot/j48_breast-cancer.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/breast-cancer.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_breast-cancer.dot


:: ------------------------------------------------ CONTACT LENSES

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/contact-lenses.arff -C 0.25 -M 2 -g > dot/j48_contact-lenses.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/contact-lenses.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_contact-lenses.dot  


:: ------------------------------------------------ PRIMARY TUMOR

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/primary-tumor.arff -C 0.25 -M 2 -g > dot/j48_primary-tumor.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/primary-tumor.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_primary-tumor.dot  


:: ------------------------------------------------ VOTE

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/vote.arff -C 0.25 -M 2 -g > dot/j48_vote.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/vote.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_vote.dot 


:: ------------------------------------------------ WEATHER

:: J48

java -cp weka.jar weka.classifiers.trees.J48 -t dataset/weather.nominal.arff -C 0.25 -M 2 -g > dot/j48_weather.dot

:: MIT

java -cp weka.jar weka.classifiers.trees.MIT -t dataset/weather.nominal.arff -C 0.25 -M 128 -I 64 -g > dot/MIT_weather.dot 
