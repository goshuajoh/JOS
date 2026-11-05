# JOS

* [`s1gate/`](s1gate/README.md): a System 1 decision gate for the Classify state of the POS
  ingestion FSM. It is a small local model that returns one typed option per question with a
  calibrated probability, and escalates to `ask_human` under a confidence threshold. The folder
  has the code for all eight build steps, the labelling guide, and the training, dev and eval
  data ([`s1gate/data/README.md`](s1gate/data/README.md)).
