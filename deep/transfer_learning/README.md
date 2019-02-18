# Uso del transfer learning per il riconoscimento dello sport praticato dell'immagine

La rete è stata creata partendo da una modello già addestrato. E' utilizzata per riconoscere lo sport praticato dai soggetti presenti nell'immagine, tra sette diverse categorie: canottaggio, badminton, polo, snowboard, croquet, barca a vela e arrampicata. Il dataset è stato creato partendo dal progetto [Event Dataset](http://vision.stanford.edu/lijiali/event_dataset/) della Stanford University e arricchito facendo uso di Google immagini: è possibile scaricarlo dal seguente [link](https://drive.google.com/open?id=1CvL6ofTO37E7eN2BMY9QuWZpVliFw5ze)

1. Scaricare il dataset, scompattare il file e lasciare la cartella dentro "transfer_learning"
2. Creare il test set utilizzando creo_test_set
3. Addestrare la rete usando transfer_learning
4. E' possibile testare la rete con il test set usando test
