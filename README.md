# rank-reservation-choices
Spark experipent to rank reservation by customer behaviour


# Automatic reservation (Ranking proposal)

## Caso d'uso: 
il software di prenotazione medica in base ai dati inseriti propone secondo le norme di legge e logiche interne un elenco di possibili strutture mediche e giorni disponibili in cui poter ottenere la prestazione sanitaria prescritta.
L'obiettivo è aggiungere un modello di ranking alle proposte in base al comportamento del cluster a cui appartiene l'assistito.
 

## Feature rilevate per il training del modello:




## Risoluzione proposta:
ho creato una pipeline che segue i seguenti passi

1) Preprocessing:
trasformazione dei dati in numeri, quantizzando alcune feature ed applicando one hot encoding su caratteristiche che corrispondono a categorie (praticamente quasi tutte)

2) Data reduction:
Riduzione dei dati e dimensionalità con una rete neurale di embedding per riuscire a governare la molteplicità delle feature. (step non ancora implementato)
Sto studiando questo paper "Deep Embedding Network for Clustering" (vedi allegato PHH2014ICPR.pdf) è la strada giusta?

3) Clustering:
Applicazione di un kmeans per clusterizzare i vari sample, da decidere nei primi testi abbiamo avuto risultati interessanti con 40 cluster.

4) Calcolo frequenza valori per le scelte del cluster:
Per ogni cluster prendo le feature delle proposte dal motore di prenotazione che sono state scelte e ne calcolo la frequenza. In modo da capire le scelte più ricorrenti di ogni cluster e poter dare un ordine di preferenza.

5) Predizione nel test set:
Prendiamo i casi test set, per ogni sample fornito al modello di clustering viene assegnato un cluster di appartenenza. Deciso il cluster di appartenenza abbiamo il ranking delle preferenze.
Per ogni feature relativa alla scelta viene valutato in che posizione si trova nel ranking in modo da capire se si trova nei valori più ricorrenti.

6) Test di accuratezza:
Per ogni predizione prendo confronto il ranking proposto con le scelte realmente fatte nel sample, se la scelta fatta si trova nella prima posizione del ranking allora abbiamo il 100% dell'accuratezza, mano a mano che la scelta fatta 
esempio:
per la feature fasce di orari preferite abbiamo 4 diversi valori PRIMO\_MATTINO (00:00-11:00), SECONDA\_MATTINATA (11:00-13:00), PRIMO\_POMERIGGIO (13:00-15:00) e SERA (15:00-23:59) se nel test la scelta è stata PRIMO_MATTINO ed il ranking è 1° PRIMO_MATTINO, 2° SERA, 3° SECONDA_MATTINATA, 4° PRIMO_POMERIGGIO. dato che la scelta è al primo posto nel ranking allora avremo il 100% dell'accuratezza. Se nella scelta avessimo avuto SECONDA_MATTINATA il ranking sarebbe stato (4-2)/4 = 50% 
Come metodo nel calcolo dell'accuratezza è corretto?

La soluzione che ho implementato in questo momento è molto semplificata molto ma con lo scopo di raggiungere un risultato e poterne discutere.



#### per installare

01_update_jvm8_spark.sh


https://www.quora.com/Why-are-there-two-ML-implementations-in-Spark-ML-and-MLlib-and-what-are-their-different-features
https://stackoverflow.com/questions/42790037/pca-on-pyspark-irregular-execution
https://www.youtube.com/watch?v=IHZwWFHWa-w
https://www.quora.com/Does-it-make-sense-to-perform-principal-components-analysis-before-clustering-if-the-original-data-has-too-many-dimensions-Is-it-theoretically-unsound-to-try-to-cluster-data-with-no-correlation

http://ranger.uta.edu/~chqding/papers/KmeansPCA1.pdf
https://github.com/piiswrong/dec


https://datascience.stackexchange.com/questions/17216/pca-before-k-mean-clustering
https://stats.stackexchange.com/questions/157621/how-would-pca-help-with-a-k-means-clustering-analysis

https://github.com/piiswrong/dec



#### to clean all *.pyc

find . -name "*.pyc" -type f -delete


#### project structure

https://github.com/kennethreitz/samplemod



#### to build

http://lorenamesa.com/packaging-my-first-python-egg.html
python setup.py build
python setup.py sdist
python setup.py bdist_egg


#### code style

http://docs.python-guide.org/en/latest/
https://www.python.org/dev/peps/pep-0008/



#### TO EXECUTE PIPELINE

../spark-2.2.0-bin-hadoop2.7/bin/spark-submit --py-files rankchoices-0.1.0-py2.7.egg  rankchoices/pipeline.py 